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

slide_data = [{
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Hamamatsu/OS-1.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Hamamatsu/OS-2.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Hamamatsu/OS-3.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Aperio/CMU-1.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Aperio/CMU-2.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Leica/SCN-1.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Leica/SCN-2.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Ventana/VT-1.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/Ventana/VT-2.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}, {
    "download_url": ("https://openslide.cs.cmu.edu/download/"
                     "openslide-testdata/MIRAX/MRXS-1.ndpi"),
    "page_url":
    "https://openslide.org/demo/"
}]


def download_assets():
    """Download required assets to cache directory"""
    print("\nChecking assets in cache...")
    model_assets = [
        ("histogpt-1b-6k-pruned.pth",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/histogpt-1b-6k-pruned.pth?download=true"
         ),
        ("ctranspath.pth",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/ctranspath.pth?download=true"
         )
    ]

    # Create slide assets list
    slide_assets = []
    for slide in slide_data:
        url = slide["download_url"]
        filename = os.path.basename(url)
        slide_assets.append((filename, url))

    # Combine all assets
    assets = model_assets + slide_assets

    for filename, url in assets:
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-O", cache_path, url], check=True)
        else:
            print(f"Found in cache: {filename}")


def prepare_directories():
    """Prepare directory structure and copy cached slides"""
    os.makedirs("./content/slide_folder", exist_ok=True)
    os.makedirs("./content/save_folder", exist_ok=True)
    os.makedirs("./report", exist_ok=True)

    # Copy all slides from cache
    for slide in slide_data:
        url = slide["download_url"]
        filename = os.path.basename(url)
        cached_slide = os.path.join(CACHE_DIR, filename)
        dest_slide = os.path.join("./content/slide_folder", filename)
        if not os.path.exists(dest_slide):
            shutil.copy(cached_slide, dest_slide)
            print(f"Copied {filename} to slide folder")
        else:
            print(f"Slide {filename} already exists in slide folder")


def process_slide():
    print("\nProcessing slides...")
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
            f"Processing output directory not found: {output_dir}")

    h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
    if not h5_files:
        raise RuntimeError("No H5 files generated during processing")

    print(f"Successfully processed {len(h5_files)} slides")


def initialize_model(device):
    """Initialize HistoGPT model"""
    print("\nInitializing model...")
    histogpt = HistoGPTForCausalLM(BioGptConfig(),
                                   PerceiverResamplerConfig()).to(device)
    model_path = os.path.join(CACHE_DIR, "histogpt-1b-6k-pruned.pth")
    state_dict = torch.load(model_path, map_location=device)
    histogpt.load_state_dict(state_dict, strict=True)
    return histogpt


def generate_report(model, device, h5_path):
    """Generate diagnostic report for a single slide"""
    print(f"\nGenerating report for {os.path.basename(h5_path)}...")
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


def save_results(report_dir, report, slide_path):
    """Save results to report directory"""
    print(f"\nSaving results to {report_dir}...")

    # Save text report
    report_path = os.path.join(report_dir, "diagnosis.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Save thumbnail
    try:
        slide = OpenSlide(slide_path)
        level = slide.get_best_level_for_downsample(32)
        thumbnail = slide.read_region(
            (0, 0), level, slide.level_dimensions[level]).convert("RGB")

        plt.figure(figsize=(10, 10))
        plt.imshow(thumbnail)
        plt.axis('off')
        plt.title("Slide Thumbnail", fontsize=12)

        image_path = os.path.join(report_dir, "thumbnail.png")
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    try:
        download_assets()
        prepare_directories()
        process_slide()
        model = initialize_model(device)

        # Process generated H5 files
        h5_dir = os.path.join("./content/save_folder/h5_files",
                              "256px_ctranspath_0.0mpp_4.0xdown_normal")

        for h5_file in os.listdir(h5_dir):
            if not h5_file.endswith('.h5'):
                continue

            h5_path = os.path.join(h5_dir, h5_file)
            slide_name = os.path.splitext(h5_file)[0]
            slide_filename = f"{slide_name}.ndpi"
            slide_path = os.path.join("./content/slide_folder", slide_filename)

            if not os.path.exists(slide_path):
                print(f"Corresponding slide not found for {h5_file}, skipping")
                continue

            # Generate report
            try:
                report = generate_report(model, device, h5_path)
            except Exception as e:
                print(f"Failed to generate report for {slide_name}: {str(e)}")
                continue

            # Create report directory
            report_dir = os.path.join("./report", slide_name)
            os.makedirs(report_dir, exist_ok=True)

            # Save results
            try:
                save_results(report_dir, report, slide_path)
                print(f"\nReport for {slide_name}:")
                print(textwrap.fill(report, width=80))
            except Exception as e:
                print(f"Failed to save results for {slide_name}: {str(e)}")

    except Exception as e:
        print(f"Critical error occurred: {str(e)}")


if __name__ == "__main__":
    main()
