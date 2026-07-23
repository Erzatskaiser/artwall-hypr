# artwall-hypr

A headless, AI-powered annotation tool designed specifically for the Hyprland Wayland compositor.

`artwall-hypr` automatically detects the currently active wallpaper, identifies classical artwork using a hybrid local/online pipeline, and
non-destructively annotates the image with historical metadata and AI-generated descriptions. The typography is dynamically scaled to match the physical resolution of the active monitor and styled to integrate seamlessly with minimal desktop environments (such as a Gruvbox Waybar configuration).

## Architecture

The identification process is designed to minimize API quota usage and prevent Vision Language Model (VLM) hallucinations. It operates on a strict four-tier pipeline:

1. **SHA-256 Cache:** Identical files are instantly retrieved from local JSON storage to prevent redundant processing.
2. **Local CLIP Vector Search:** A lightweight `ViT-B-32` model runs on the system CPU to perform a rapid cosine-similarity search against a local database of museum artworks. This preserves 100% of GPU VRAM for the Ollama inference engine.
3. **SerpApi Fallback:** If the local index yields no high-confidence matches, the image is temporarily uploaded and processed via Google Lens reverse-image search to extract factual titles.
4. **Haiku Fallback Mode:** If online identification fails or network connectivity is lost, the system actively prohibits the local VLM from guessing historical facts. Instead, it generates a structured 3-line haiku inspired by the visual mood of the painting.

Once metadata is secured, the system calls a local Ollama instance to generate a concise visual description. The final text is rendered onto a temporary file using ImageMagick and pushed live via `hyprpaper` IPC.

## Prerequisites

### System Requirements

* **OS:** Linux
* **Compositor:** Hyprland
* **Wallpaper Daemon:** `hyprpaper`
* **Dependencies:** `imagemagick` (for the `magick` command)
* **AI Engine:** `ollama` running locally with a vision model (e.g., `llava:7b` or `moondream`)

### Python Requirements

* Python 3.10+
* A modern Python package manager (such as `uv` or `poetry`)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/artwall-hypr.git
cd artwall-hypr

```


2. **Set up the virtual environment and install dependencies:**
Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
uv sync

```


3. **Configure the system:**
Review `artwall/config.py` to ensure the `wallpaper_dir` matches your local gallery path. You can also adjust typography multipliers, fonts, and colors to match your desktop aesthetic. By default, it uses a Gruvbox dark background with light text.
4. **Set environment variables (Optional but recommended):**
```bash
export SERPAPI_KEY="your_serpapi_key_here"
export ARTWALL_VLM_MODEL="llava:7b"

```



## Usage

### 1. Populate the Local Gallery Index

Before running the main script, it is highly recommended to populate your local CLIP vector index. This script securely streams public domain artwork from the Art Institute of Chicago, The Metropolitan Museum of Art, and WikiArt to build a local `.npy` vector database.

```bash
python populate_gallery_index.py

```

### 2. Run the Annotator

Once your wallpaper is set via `hyprpaper`, execute the main script to process the image and apply the annotation.

```bash
python main.py

```

To view detailed execution logs, including CLIP similarity scores and HTTP requests, append the verbose flag:

```bash
python main.py -v

```

## How the Dynamic Rendering Works

To prevent unexpected cropping on different monitor aspect ratios (e.g., switching between a 16:9 laptop screen and a 21:9 ultrawide monitor), `artwall-hypr` queries `hyprctl` to determine the exact physical dimensions of the active display.

ImageMagick pre-crops the artwork to match these dimensions identically before calculating typography constraints. Font sizes and margins are calculated as strict percentages of the screen height and width, ensuring the annotation block maintains the exact same visual weight across all devices without manual configuration.
