# artwall-hypr

An offline, local-AI desktop widget and annotation engine designed specifically for the Hyprland Wayland compositor.

`artwall-hypr` automatically detects your active wallpaper, identifies classical artwork using a 100% local vector search engine, and projects a sleek, Waybar-aligned GTK Layer-Shell overlay over your desktop. It provides historical metadata, confidence-rated artist attribution, and AI-generated visual descriptions without mutating your original wallpaper files or sending image data to external cloud APIs.

---

## Key Features

* **100% Offline & Local-First:** No cloud API keys, no monthly fees, and zero web scrapers.
* **GTK Layer-Shell Wayland Overlay:** Replaces heavy image-editing tools with a native, GPU-accelerated transparent GTK widget that sits directly above your wallpaper layer.
* **OpenCLIP Vector Search:** Fast offline vector matching powered by PyTorch and OpenAI's CLIP (`ViT-B-32`) to identify masterpieces from your local database in milliseconds.
* **Anti-Hallucination Guardrails:** Utilizes local Ollama VLMs (`llava`, `moondream`) with strict prompt constraints to write museum curator blurbs or evocative haikus without spatial rambling.
* **Graduated Confidence Modes:** Gracefully indicates certainty levels (e.g., *Reminiscent of...*) when image matches fall between fuzzy and exact thresholds.
* **Automated Daemon Mode:** Includes a lightweight background listener (`daemon.py`) that detects wallpaper updates instantly across any randomizer script or keybinding.

---

## Architecture

The system operates on a strict **local-first 3-tier pipeline** designed to maximize speed while preserving system VRAM:

```
[ Active Wallpaper ]
         │
         ▼
 ┌───────────────┐
 │ SHA-256 Cache │ ──( Hit )──► [ Instant Metadata Overlay ]
 └───────┬───────┘
         │ ( Miss )
         ▼
 ┌──────────────────────┐
 │ Local OpenCLIP Index │ ──( Cosine Match )──► Score ≥ 0.92 : Confident Curator Blurb
 └───────┬──────────────┘                       Score ≥ 0.75 : "Reminiscent of..." Mode
         │ ( Miss )
         ▼
 ┌──────────────────────┐
 │ Local Ollama VLM     │ ──► Configurable Fallback ('reminiscent' or 'haiku')
 └──────────────────────┘

```

1. **SHA-256 Cache:** Hashes the active wallpaper file to instantly restore previously processed metadata from local storage.
2. **Local OpenCLIP Vector Search:** Computes 512-dimensional vector embeddings on the system CPU and runs Cosine Similarity searches against a pre-computed matrix (`.npy`) of museum artworks.
3. **Ollama VLM Processing:**
* **Exact Match ($\ge 0.92$):** Generates an educational 2-sentence museum curator description.
* **Fuzzy Match ($0.75 \le \text{Score} < 0.92$):** Updates the UI title to *Reminiscent of...* and prompts the VLM to compare stylistic techniques.
* **Vector Miss ($< 0.75$):** Triggered based on your `config.py` setting—either forcing a comparison to the nearest database artwork or writing a 3-line haiku.



---

## Prerequisites

### System Requirements

* **OS:** Linux (Arch, Fedora, Debian, etc.)
* **Compositor:** Hyprland
* **Wallpaper Daemon:** `hyprpaper` (or any wallpaper changer)
* **System Libraries:**
* `gtk3`
* `gtk-layer-shell`
* `gobject-introspection`
* `cairo`
* `pkgconf`


* **AI Engine:** `ollama` running locally with a vision model (e.g., `llava:7b` or `moondream`)

#### Installing System Dependencies (Arch Linux)

```bash
sudo pacman -S gtk3 gtk-layer-shell gobject-introspection cairo pkgconf

```

### Python Requirements

* Python 3.10+
* Package Manager: `uv` (recommended) or standard `pip`

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/artwall-hypr.git
cd artwall-hypr

```


2. **Set up the virtual environment and install dependencies:**
Using `uv`:
```bash
uv venv
source .venv/bin/activate
uv sync
uv add PyGObject

```


3. **Configure the system:**
Review `artwall/config.py` to ensure `wallpaper_dir` points to your local wallpaper collection:
```python
# Fallback behavior when vector search confidence is low ('reminiscent' or 'haiku')
fallback_mode: str = "reminiscent"

```



---

## Usage

### 1. Populate the Local Gallery Index

Build your local vector database before launching the annotator. This script streams public domain artwork metadata from major institutions (such as the Art Institute of Chicago and WikiArt) to construct your local `.npy` vector index.

```bash
python populate_gallery_index.py

```

### 2. Run the Background Daemon (Recommended)

To make `artwall-hypr` run autonomously, start the daemon process. It sits silently in the background using negligible CPU, listening for wallpaper swaps:

```bash
python -m artwall.daemon

```

To auto-start the daemon on login, add this line to your `~/.config/hypr/hyprland.conf`:

```text
exec-once = ~/.venv/bin/python -m /path/to/artwall-hypr/artwall/daemon.py

```

### 3. Run One-Shot Execution (CLI)

If you prefer to trigger the analysis manually via keybindings or scripts:

```bash
python main.py

```

Pass the verbose flag for real-time debug logs and CLIP similarity scores:

```bash
python main.py -v

```

---

## How the GTK Wayland Overlay Works

Unlike legacy scripts that burn text directly into wallpaper JPEGs, `artwall-hypr` uses **`gtk-layer-shell`** to construct a native GTK3 surface anchored to the `BOTTOM` Wayland layer.

* **Zero File Mutation:** Your original high-resolution wallpaper files remain completely untouched on disk and in VRAM.
* **Waybar Geometry Alignment:** Margin offsets (`10px`) and typography scales (`14px JetBrains Mono`) are synchronized with Waybar's `.modules-right` rules for visual cohesion.
* **Pango Vector Typography:** Text is rendered natively by the GTK font engine with crisp word-wrapping, right-justification, and rounded container borders (`border-radius: 10px`).
