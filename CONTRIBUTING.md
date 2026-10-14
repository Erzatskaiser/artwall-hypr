# Contributing

## Setup

```bash
uv sync --extra dev
```

PyGObject builds against system GTK libraries, so install those first. On Arch:

```bash
sudo pacman -S gtk3 gtk-layer-shell gobject-introspection cairo pkgconf
```

## Tests

```bash
uv run pytest tests/
```

The suite avoids torch and GTK so it stays fast, and tests needing PyGObject skip
themselves when it is missing. Anything touching path handling or the overlay markup
should come with a test, since those are the two places that have broken before.

## Working on it

Running the pipeline by hand is the quickest way to see a change:

```bash
uv run artwall -v        # scores, timings, and which tier was taken
uv run artwall --clear   # take the overlay off screen
```

A full index build takes hours. For development, a few hundred artworks is plenty:

```bash
uv run artwall-index --sources met --limit 200
```

## Notes

* The CLIP model runs on CPU on purpose, so it does not compete with Ollama for VRAM.
* Keep the overlay process free of heavy imports. It is spawned on every wallpaper change
  and should appear quickly.
* Anything written into Pango markup must be escaped. An unescaped `&` in an artwork
  title makes the whole overlay render blank.
