import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Paths
    wallpaper_dir: Path = Path.home() / "Images" / "Wallpapers" / "paintings"
    temp_rendered_path: Path = Path("/tmp/artwall_current.jpg")
    
    # Ollama Settings
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    vlm_model: str = os.getenv("ARTWALL_VLM_MODEL", "llava:7b")
    
    # Typography & Styling
    font: str = "/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf"
    fill_color: str = "#ebdbb2"    # Gruvbox light text
    box_color: str = "#282828"     # Gruvbox dark background
    stroke_color: str = "none"
    stroke_width: int = 0
    
    interline_spacing: int = 6
    gravity: str = "SouthEast"
    
    # DYNAMIC SCALING MULTIPLIERS (Relative to active monitor resolution)
    font_size_pct: float = 0.018
    margin_x_pct: float = 0.025       
    margin_y_pct: float = 0.035
