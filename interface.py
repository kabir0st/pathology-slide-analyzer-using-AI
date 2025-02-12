import os
import shutil
import subprocess
import textwrap
import uuid

import h5py
import requests
import streamlit as st
import torch
from openslide import OpenSlide
from transformers import BioGptConfig, BioGptTokenizer

from histogpt.helpers.inference import generate
from histogpt.helpers.patching import PatchingConfigs
from histogpt.helpers.patching import main as patching_main
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig

# Set page configuration
st.set_page_config(page_title="Slide Analyzer", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .metric-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1) }
    .stProgress > div > div > div > div { background: #1e88e5 }
</style>""",
            unsafe_allow_html=True)

# Define cache directory
CACHE_DIR = os.path.expanduser(".cache/histogpt")
os.makedirs(CACHE_DIR, exist_ok=True)


def download_assets():
    assets = [
        ("histogpt-1b-6k-pruned.pth",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/histogpt-1b-6k-pruned.pth"
         ),
        ("ctranspath.pth",
         "https://huggingface.co/marr-peng-lab/histogpt/resolve/main/ctranspath.pth"
         )
    ]

    for filename, url in assets:
        cache_path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            with st.spinner(f"Downloading {filename}..."):
                subprocess.run(["wget", "-O", cache_path, url])


@st.cache_resource
def initialize_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    histogpt = HistoGPTForCausalLM(BioGptConfig(),
                                   PerceiverResamplerConfig()).to(device)
    model_path = os.path.join(CACHE_DIR, "histogpt-1b-6k-pruned.pth")
    histogpt.load_state_dict(torch.load(model_path, map_location=device),
                             strict=True)
    return histogpt, device


def process_slide(slide_path, session_id):
    # Create session-specific directories
    slide_dir = f"./content/slide_folder_{session_id}"
    save_dir = f"./content/save_folder_{session_id}"
    os.makedirs(slide_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Copy slide to processing directory
    shutil.copy(slide_path,
                os.path.join(slide_dir, os.path.basename(slide_path)))

    # Configure processing
    configs = PatchingConfigs()
    configs.slide_path = slide_dir
    configs.save_path = save_dir
    configs.model_path = os.path.join(CACHE_DIR, 'ctranspath.pth')
    configs.patch_size = 256
    configs.white_thresh = [170, 185, 175]
    configs.edge_threshold = 2
    configs.resolution_in_mpp = 0.0
    configs.downscaling_factor = 4.0
    configs.batch_size = 16

    with st.spinner("Processing slide features..."):
        patching_main(configs)

    return os.path.join(save_dir, "h5_files",
                        "256px_ctranspath_0.0mpp_4.0xdown_normal")


def generate_report(model, device, h5_path):
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    prompt = 'Final diagnosis:'
    prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

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


# App title
st.title("🔬 HistoGPT Slide Analyzer")
st.markdown("---")

# Download required assets
download_assets()


def download_file(file_url, filename):
    file_response = requests.get(file_url, stream=True)
    file_response.raise_for_status()

    total_size = int(file_response.headers.get('content-length', 0))
    downloaded_size = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    with open(filename, 'wb') as f:
        for file_chunk in file_response.iter_content(chunk_size=8192):
            if file_chunk:
                f.write(file_chunk)
                downloaded_size += len(file_chunk)
                if total_size > 0:
                    progress = downloaded_size / total_size
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Downloaded"
                                     f" {downloaded_size/1024/1024:.1f} "
                                     f"MB of {total_size/1024/1024:.1f} MB")

    progress_bar.empty()
    status_text.empty()
    return filename


# URL input
url = st.text_input("Enter NDPI file URL:",
                    placeholder="https://example.com/path/to/slide.ndpi")

if url:
    session_id = uuid.uuid4().hex
    slide_path = f"temp_slide_{session_id}.ndpi"
    model, device = initialize_model()

    try:
        # Download slide
        with st.spinner("Downloading slide..."):
            download_file(url, slide_path)
            st.success("✅ Slide downloaded")

        # Process slide
        h5_dir = process_slide(slide_path, session_id)
        h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]

        if not h5_files:
            st.error("No H5 files generated during processing")
            raise RuntimeError("H5 file generation failed")

        # Generate report
        with st.spinner("Generating diagnosis..."):
            h5_path = os.path.join(h5_dir, h5_files[0])
            report = generate_report(model, device, h5_path)

        # Display results
        st.markdown("---")
        st.subheader("Diagnostic Report")
        st.markdown(f"```\n{textwrap.fill(report, width=80)}\n```")

        # Show thumbnail
        with st.spinner("Generating thumbnail..."):
            slide = OpenSlide(slide_path)
            level = slide.get_best_level_for_downsample(32)
            thumbnail = slide.read_region(
                (0, 0), level, slide.level_dimensions[level]).convert("RGB")

            st.subheader("Slide Thumbnail")
            st.image(thumbnail, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(slide_path):
            os.remove(slide_path)
        shutil.rmtree(f"./content/slide_folder_{session_id}",
                      ignore_errors=True)
        shutil.rmtree(f"./content/save_folder_{session_id}",
                      ignore_errors=True)

else:
    st.info("ℹ️ Please enter a valid NDPI URL to begin analysis")
    st.markdown("---")
    st.subheader("Example Usage:")
    st.write("1. Enter a valid NDPI file URL in the input field above")
    st.write("2. The file will be downloaded and processed automatically")
    st.write("3. AI-generated diagnostic report will be displayed")
