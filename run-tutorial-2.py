import os
import shutil
import textwrap
import subprocess
import torch
import h5py
import matplotlib.pyplot as plt
from openslide import OpenSlide
from transformers import BioGptConfig, BioGptTokenizer
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
from histogpt.helpers.patching import main as patching_main, PatchingConfigs
from histogpt.helpers.inference import generate


def setup_environment():
    """Install required dependencies and setup environment"""
    print("Setting up environment...")
    subprocess.run(["sudo", "apt-get", "install", "openslide-tools", "-y"])
    subprocess.run(["sudo", "apt-get", "install", "python-openslide", "-y"])
    subprocess.run([
        "pip", "install", "openslide-python", "flamingo-pytorch",
        "git+https://github.com/marrlab/HistoGPT"
    ])


def download_assets():
    """Download required model weights and example slide"""
    print("\nDownloading assets...")
    subprocess.run([
        "wget", "-O", "histogpt-1b-6k-pruned.pth",
        "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/histogpt-1b-6k-pruned.pth?download=true"
    ])
    subprocess.run([
        "wget", "-O", "ctranspath.pth",
        "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/ctranspath.pth?download=true"
    ])
    subprocess.run([
        "wget", "-O", "example_slide.ndpi",
        "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/2023-03-06%2023.51.44.ndpi?download=true"
    ])


def prepare_directories():
    """Create and organize required directories"""
    os.makedirs("/content/slide_folder", exist_ok=True)
    os.makedirs("/content/save_folder", exist_ok=True)
    shutil.move("example_slide.ndpi",
                "/content/slide_folder/2023-03-06 23.51.44.ndpi")


def process_slide():
    """Process slide and extract features using HistoGPT"""
    print("\nProcessing slide...")
    configs = PatchingConfigs()
    configs.slide_path = '/content/slide_folder'
    configs.save_path = '/content/save_folder'
    configs.model_path = 'ctranspath.pth'
    configs.patch_size = 256
    configs.white_thresh = [170, 185, 175]
    configs.edge_threshold = 2
    configs.resolution_in_mpp = 0.0
    configs.downscaling_factor = 4.0
    configs.batch_size = 16
    patching_main(configs)


def initialize_model(device):
    """Initialize and load HistoGPT model"""
    print("\nInitializing model...")
    histogpt = HistoGPTForCausalLM(BioGptConfig(),
                                   PerceiverResamplerConfig()).to(device)
    state_dict = torch.load("histogpt-1b-6k-pruned.pth", map_location=device)
    histogpt.load_state_dict(state_dict, strict=True)
    return histogpt


def generate_report(model, device):
    """Generate diagnostic report using the model"""
    print("\nGenerating report...")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    prompt = 'Final diagnosis:'
    prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with h5py.File(
            '/content/save_folder/h5_files/256px_ctranspath_0.0mpp_4.0xdown_normal/2023-03-06 23.51.44.h5',
            'r') as f:
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


def display_results():
    """Display slide thumbnail and generated report"""
    print("\nDisplaying results...")
    # Show slide thumbnail
    slide = OpenSlide('/content/slide_folder/2023-03-06 23.51.44.ndpi')
    level = slide.get_best_level_for_downsample(32)
    thumbnail = slide.read_region((0, 0), level,
                                  slide.level_dimensions[level]).convert("RGB")

    plt.figure(figsize=(10, 10))
    plt.imshow(thumbnail)
    plt.axis('off')
    plt.title("Slide Thumbnail", fontsize=12)
    plt.show()


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    try:
        setup_environment()
        download_assets()
        prepare_directories()
        process_slide()
        model = initialize_model(device)
        report = generate_report(model, device)

        # Display results
        display_results()
        print("\nGenerated Report:")
        print(textwrap.fill(report, width=80))

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
