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
    
    # Fallback behavior ('haiku' or 'reminiscent')
    fallback_mode: str = "reminiscent"

    # Typography & Styling
    font: str = "/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf"
    font_size: int = 14         
    fill_color: str = "#ebdbb2"    # Gruvbox light text
    box_color: str = "#282828"     # Gruvbox dark background
    
    stroke_color: str = "none"
    stroke_width: int = 0
    gravity: str = "SouthEast"
    
    # LAYOUT MARGINS
    margin_x: int = 10             # 10px right margin
    margin_y_pct: float = 0.035    # Bottom margin
