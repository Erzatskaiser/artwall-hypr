# artwall-hypr

Identifies the painting on your Hyprland desktop and annotates it with a small overlay in
the corner of the screen. Everything runs locally: a CLIP index of museum collections does
the matching, and a local Ollama vision model writes the description. No API keys, and no
image leaves the machine.

Wallpaper files are never modified. The text is drawn on a `gtk-layer-shell` surface above
the wallpaper, so the original image stays untouched on disk.

## How it works

```
[ active wallpaper ]
         |
         v
 +---------------+
 | SHA-256 cache | --(hit)--> show stored metadata, no inference
 +-------+-------+
         | (miss)
         v
 +----------------------+
 | local CLIP index     | --> score >= 0.92 : reported as that artwork
 +-------+--------------+     score >= 0.75 : reported as "Reminiscent of ..."
         |
         v
 +----------------------+
 | local Ollama VLM     | --> writes the description, or a haiku on a weak match
 +----------------------+
```

1. The wallpaper file is hashed. A hash seen before skips straight to the stored result,
   so repeat wallpapers cost nothing.
2. CLIP (`ViT-B-32`, on CPU) embeds the image and compares it against a prebuilt matrix of
   museum artwork embeddings using cosine similarity.
3. The score decides how the result is phrased. A confident match is named outright, a
   fuzzy one is described as a resemblance, and a weak match falls back to either a
   resemblance or a haiku depending on `fallback_mode`.
4. The vision model writes one or two sentences, which go into the overlay.

Thresholds and fallback behaviour live in `artwall/config.py`.

## Requirements

* Hyprland, with `hyprpaper` setting the wallpaper
* `ollama` running locally with a vision model pulled, such as `llava:7b` or `moondream`
* Python 3.14 or newer
* System libraries: `gtk3`, `gtk-layer-shell`, `gobject-introspection`, `cairo`, `pkgconf`

On Arch:

```bash
sudo pacman -S gtk3 gtk-layer-shell gobject-introspection cairo pkgconf
ollama pull llava:7b
```

## Installation

```bash
git clone https://github.com/Erzatskaiser/artwall-hypr.git
cd artwall-hypr
uv sync
```

That creates the virtualenv, installs the dependencies, and puts `artwall`,
`artwall-daemon` and `artwall-index` on the path inside it. PyGObject builds against the
system GTK libraries listed above, so install those first.

Without `uv`, a plain `python -m venv .venv && pip install -e .` works the same way.

## Building the index

Nothing can be identified until the local index exists. This downloads thumbnails from
open access museum APIs, embeds them, and writes the vectors to
`~/.cache/artwall/clip_index`:

```bash
uv run artwall-index
```

Sources are the National Gallery of Art, the Cleveland Museum of Art, the National Gallery
of Denmark, the Art Institute of Chicago, WikiArt, and the Met. A full run fetches roughly
15,000 artworks and takes a few hours, mostly waiting on rate limits. It can be
interrupted, and whatever has been indexed so far is saved.

For a subset, or a shorter run:

```bash
uv run artwall-index --sources nga,met --limit 500
```

## Running it

Once, against the current wallpaper:

```bash
uv run artwall          # -v shows match scores and timings
uv run artwall --clear  # take the overlay off screen
```

As a daemon that watches for wallpaper changes:

```bash
uv run artwall-daemon
```

To start it with your session, add this to `~/.config/hypr/hyprland.conf`:

```text
exec-once = /path/to/artwall-hypr/.venv/bin/python -m artwall.daemon
```

Inside an activated virtualenv the same commands work without the `uv run` prefix.

## Configuration

`artwall/config.py` holds the settings. The ones worth looking at first:

| Setting | Default | Meaning |
|---|---|---|
| `wallpaper_dir` | `~/Images/Wallpapers/paintings` | Wallpapers outside this directory are ignored |
| `vlm_model` | `llava:7b` | Any vision model available to Ollama |
| `fallback_mode` | `reminiscent` | What a weak match produces: `reminiscent` or `haiku` |
| `confident_score` | `0.92` | At or above this, the artwork is named outright |
| `fuzzy_score` | `0.75` | At or above this, it is reported as a resemblance |
| `font_family`, `fg_color`, `bg_color` | Gruvbox | Overlay styling |
| `margin_x`, `margin_y`, `width_chars` | 10, 10, 45 | Overlay position and text width |

`OLLAMA_URL` and `ARTWALL_VLM_MODEL` can also be set in the environment.

The wallpaper directory is compared after resolving symlinks, so pointing
`~/Images/Wallpapers` at a synced folder elsewhere works.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/
```

## Limitations

* CLIP matches the image, so a cropped, recoloured or heavily filtered version of a
  painting can score below the confident threshold even when the work is in the index.
* The index only covers what the ingest sources publish. Anything outside it falls back to
  a resemblance or a haiku.
* Descriptions come from a small local vision model. They read well enough, but they are
  not art history.
* Inference takes anywhere from a few seconds to a minute depending on model and hardware,
  so the overlay appears shortly after the wallpaper changes. Cached wallpapers are
  instant.
* The daemon polls `hyprctl` instead of subscribing to an event, because hyprpaper does
  not emit wallpaper change events.
* The overlay sits on the bottom layer, just above the wallpaper, so any open window
  covers it. It is meant to be read on an empty workspace.

## License

GPL-3.0. See [LICENSE](LICENSE).
