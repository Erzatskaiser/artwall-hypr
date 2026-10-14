import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Runtime settings. Paths are resolved so symlinked wallpaper directories work."""

    # Where your art wallpapers live. Wallpapers outside this directory are ignored.
    wallpaper_dir: Path = Path.home() / "Images" / "Wallpapers" / "paintings"

    # Local state: metadata cache, overlay state file, and the CLIP vector index.
    cache_dir: Path = Path.home() / ".cache" / "artwall"
    index_dir: Path = Path.home() / ".cache" / "artwall" / "clip_index"

    # Ollama
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    vlm_model: str = os.getenv("ARTWALL_VLM_MODEL", "llava:7b")
    ollama_timeout: int = 120

    # Cosine similarity thresholds for the CLIP index. At or above confident_score the
    # match is reported as the artwork itself; between fuzzy_score and confident_score it
    # is reported as a resemblance.
    confident_score: float = 0.92
    fuzzy_score: float = 0.75

    # What happens below fuzzy_score: "haiku" or "reminiscent".
    fallback_mode: str = "reminiscent"

    # Overlay appearance. Defaults line up with a Gruvbox Waybar.
    font_family: str = "JetBrainsMono Nerd Font Mono"
    font_size: int = 14
    fg_color: str = "#ebdbb2"
    bg_color: str = "#282828"
    corner_radius: int = 10
    margin_x: int = 10
    margin_y: int = 10
    width_chars: int = 45

    def __post_init__(self):
        self.wallpaper_dir = Path(self.wallpaper_dir).expanduser()
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.index_dir = Path(self.index_dir).expanduser()

    @property
    def state_file(self) -> Path:
        """Metadata handed to the overlay process."""
        return self.cache_dir / "current_art.json"

    @property
    def cache_file(self) -> Path:
        """Hash to metadata mapping for wallpapers already processed."""
        return self.cache_dir / "known_art.json"

    def is_art_wallpaper(self, path: Path) -> bool:
        """
        True if path sits inside wallpaper_dir.

        Both sides are resolved first, so a symlinked wallpaper directory (for example
        ~/Images/Wallpapers pointing into a cloud-synced folder) still matches.
        """
        try:
            resolved = Path(path).resolve()
            root = self.wallpaper_dir.resolve()
        except OSError:
            return False
        return root == resolved.parent or root in resolved.parents
