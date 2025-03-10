import os
import shutil
import subprocess
import textwrap
import requests

import h5py
import matplotlib.pyplot as plt
import torch
from openslide import OpenSlide
from transformers import BioGptConfig, BioGptTokenizer

from histogpt.helpers.inference import generate
from histogpt.helpers.patching import PatchingConfigs
from histogpt.helpers.patching import main as patching_main
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig

# Define cache directory
CACHE_DIR = os.path.expanduser(".cache/histogpt")
os.makedirs(CACHE_DIR, exist_ok=True)

# Define NDPI cache directory
NDPI_CACHE_DIR = os.path.join(CACHE_DIR, "ndpi_files")
os.makedirs(NDPI_CACHE_DIR, exist_ok=True)


def download_assets():
    """Download required assets to cache directory if not exists"""
    print("\nChecking assets in cache...")
    assets = [
        (
            "histogpt-1b-6k-pruned.pth",
            "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/"
            "histogpt-1b-6k-pruned.pth?download=true"
        ),
        (
            "ctranspath.pth",
            "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/"
            "ctranspath.pth?download=true"
        ),
    ]

    for filename, url in assets:
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-O", cache_path, url])
        else:
            print(f"Found in cache: {filename}")


def download_ndpi(url, cache_dir=NDPI_CACHE_DIR):
    """Download NDPI file and cache it"""
    filename = os.path.basename(url)
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Found in cache: {filename}")
    return cache_path


def prepare_directories():
    """Prepare directory structure"""
    os.makedirs("./content/slide_folder", exist_ok=True)
    os.makedirs("./content/save_folder", exist_ok=True)
    os.makedirs("./report", exist_ok=True)  # Create base report directory


def get_next_report_number():
    """Get next sequential report number"""
    existing = [
        d for d in os.listdir("./report") if
        os.path.isdir(os.path.join("./report", d)) and d.startswith("report-")
    ]
    numbers = []
    for d in existing:
        try:
            numbers.append(int(d.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers) + 1 if numbers else 1


def process_slide(slide_path, save_path, model_path):
    print("\nProcessing slide...")
    configs = PatchingConfigs()
    configs.slide_path = slide_path
    configs.save_path = save_path
    configs.model_path = model_path
    configs.patch_size = 256
    configs.white_thresh = [170, 185, 175]
    configs.edge_threshold = 2
    configs.resolution_in_mpp = 0.0
    configs.downscaling_factor = 4.0
    configs.batch_size = 16

    try:
        patching_main(configs)
    except Exception as e:
        raise RuntimeError(f"Slide processing failed: {str(e)}")

    output_dir = os.path.join(
        configs.save_path, "h5_files", "256px_ctranspath_0.0mpp_4.0xdown_normal"
    )

    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Processing output directory not created: {output_dir}"
        )

    h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
    if not h5_files:
        raise RuntimeError("No H5 files generated during processing")

    print(f"Successfully generated {len(h5_files)} H5 files")
    return output_dir


def initialize_model(device):
    """Initialize model using cached HistoGPT weights"""
    print("\nInitializing model...")
    histogpt = HistoGPTForCausalLM(BioGptConfig(),
                                   PerceiverResamplerConfig()).to(device)
    model_path = os.path.join(CACHE_DIR, "histogpt-1b-6k-pruned.pth")
    state_dict = torch.load(model_path, map_location=device)
    histogpt.load_state_dict(state_dict, strict=True)
    return histogpt


def generate_report(model, device, h5_path):
    """Generate diagnostic report using the model"""
    print("\nGenerating report...")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    prompt = 'Final diagnosis:'
    prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 features file not found at {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        features = torch.tensor(f['feats'][:]).unsqueeze(0).to(device)

    output = generate(model=model,
                      prompt=prompt,
                      image=features,
                      length=256,
                      top_k=40,
                      top_p=0.95,
                      temp=0.7,
                      device=device)
    return tokenizer.decode(output[0, 1:])


def save_results(report_dir, report, source_url):
    """Save results to report directory"""
    print(f"\nSaving results to {report_dir}...")

    # Save text report
    report_path = os.path.join(report_dir, "diagnosis.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Save source URL
    source_path = os.path.join(report_dir, "source.txt")
    with open(source_path, 'w') as f:
        f.write(source_url)

    # Save thumbnail image
    slide_path = os.path.join("./content/slide_folder", os.path.basename(source_url))
    slide = OpenSlide(slide_path)
    level = slide.get_best_level_for_downsample(32)
    thumbnail = slide.read_region((0, 0), level,
                                  slide.level_dimensions[level]).convert("RGB")

    plt.figure(figsize=(10, 10))
    plt.imshow(thumbnail)
    plt.axis('off')
    plt.title("Slide Thumbnail", fontsize=12)

    image_path = os.path.join(report_dir, "thumbnail.png")
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

    print(f"Saved report, source, and thumbnail to {report_dir}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    try:
        download_assets()
        prepare_directories()
        model = initialize_model(device)

        import sources
        for slide in sources.slide_data:
            download_url = slide["download_url"]
            page_url = slide["page_url"]

            # Download NDPI file
            ndpi_path = download_ndpi(download_url)

            # Copy slide to slide folder
            slide_folder = "./content/slide_folder"
            dest_slide = os.path.join(slide_folder, os.path.basename(ndpi_path))
            if not os.path.exists(dest_slide):
                shutil.copy(ndpi_path, dest_slide)

            # Process slide
            save_path = "./content/save_folder"
            model_path = os.path.join(CACHE_DIR, 'ctranspath.pth')
            output_dir = process_slide(slide_folder, save_path, model_path)

            # Generate report
            h5_path = os.path.join(
                output_dir, "2023-03-06 23.51.44.h5"
            )
            report = generate_report(model, device, h5_path)

            # Create report directory
            report_number = get_next_report_number()
            report_dir = os.path.join("./report", f"report-{report_number}")
            os.makedirs(report_dir, exist_ok=True)

            # Save results
            save_results(report_dir, report, page_url)

            # Display results
            print("\nGenerated Report:")
            print(textwrap.fill(report, width=80))
            plt.imshow(plt.imread(os.path.join(report_dir, "thumbnail.png")))

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
