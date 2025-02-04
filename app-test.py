import os
import shutil
import subprocess
import textwrap

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


def download_assets():
    """Download required assets to cache directory if not exists"""
    print("\nChecking assets in cache...")
    assets = [
        ("histogpt-1b-6k-pruned.pth",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/histogpt-1b-6k-pruned.pth?download=true"
         ),
        ("ctranspath.pth",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/ctranspath.pth?download=true"
         ),
        ("example_slide.ndpi",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/2023-03-06%2023.51.44.ndpi?download=true"
         )
    ]

    for filename, url in assets:
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-O", cache_path, url])
        else:
            print(f"Found in cache: {filename}")


def prepare_directories():
    """Prepare directory structure and copy cached slide"""
    os.makedirs("./content/slide_folder", exist_ok=True)
    os.makedirs("./content/save_folder", exist_ok=True)
    os.makedirs("./report", exist_ok=True)  # Create base report directory

    # Copy slide from cache
    cached_slide = os.path.join(CACHE_DIR, "example_slide.ndpi")
    dest_slide = "./content/slide_folder/2023-03-06 23.51.44.ndpi"
    if not os.path.exists(dest_slide):
        shutil.copy(cached_slide, dest_slide)


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


def process_slide():
    print("\nProcessing slide...")
    configs = PatchingConfigs()
    configs.slide_path = './content/slide_folder'
    configs.save_path = './content/save_folder'
    configs.model_path = os.path.join(CACHE_DIR, 'ctranspath.pth')
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

    output_dir = os.path.join(configs.save_path, "h5_files",
                              "256px_ctranspath_0.0mpp_4.0xdown_normal")

    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Processing output directory not created: {output_dir}")

    h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
    if not h5_files:
        raise RuntimeError("No H5 files generated during processing")

    print(f"Successfully generated {len(h5_files)} H5 files")


def initialize_model(device):
    """Initialize model using cached HistoGPT weights"""
    print("\nInitializing model...")
    histogpt = HistoGPTForCausalLM(BioGptConfig(),
                                   PerceiverResamplerConfig()).to(device)
    model_path = os.path.join(CACHE_DIR, "histogpt-1b-6k-pruned.pth")
    state_dict = torch.load(model_path, map_location=device)
    histogpt.load_state_dict(state_dict, strict=True)
    return histogpt


def generate_report(model, device):
    """Generate diagnostic report using the model"""
    print("\nGenerating report...")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    prompt = 'Final diagnosis:'
    prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    h5_path = os.path.join("./content/save_folder/h5_files",
                           "256px_ctranspath_0.0mpp_4.0xdown_normal",
                           "2023-03-06 23.51.44.h5")

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


def save_results(report_dir, report):
    """Save results to report directory"""
    print(f"\nSaving results to {report_dir}...")

    # Save text report
    report_path = os.path.join(report_dir, "diagnosis.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Save thumbnail image
    slide = OpenSlide('./content/slide_folder/2023-03-06 23.51.44.ndpi')
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

    print(f"Saved report and thumbnail to {report_dir}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    try:
        download_assets()
        prepare_directories()
        process_slide()
        model = initialize_model(device)
        report = generate_report(model, device)

        # Create report directory
        report_number = get_next_report_number()
        report_dir = os.path.join("./report", f"report-{report_number}")
        os.makedirs(report_dir, exist_ok=True)

        # Save results
        save_results(report_dir, report)

        # Display results
        print("\nGenerated Report:")
        print(textwrap.fill(report, width=80))
        plt.imshow(plt.imread(os.path.join(report_dir, "thumbnail.png")))

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
