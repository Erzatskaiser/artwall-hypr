from pathlib import Path

from artwall.config import Config


def test_wallpaper_inside_directory_matches(tmp_path):
    art = tmp_path / "paintings"
    art.mkdir()
    wallpaper = art / "wlpp1.jpg"
    wallpaper.touch()

    cfg = Config(wallpaper_dir=art)
    assert cfg.is_art_wallpaper(wallpaper)


def test_wallpaper_outside_directory_is_rejected(tmp_path):
    art = tmp_path / "paintings"
    art.mkdir()
    other = tmp_path / "screenshots"
    other.mkdir()
    wallpaper = other / "shot.png"
    wallpaper.touch()

    cfg = Config(wallpaper_dir=art)
    assert not cfg.is_art_wallpaper(wallpaper)


def test_symlinked_wallpaper_directory_still_matches(tmp_path):
    """
    The case that broke in practice: wallpapers moved to cloud storage and the configured
    directory became a symlink, so the reported path no longer matched the configured one.
    """
    real = tmp_path / "Dropbox" / "Wallpapers" / "paintings"
    real.mkdir(parents=True)
    wallpaper = real / "wlpp43.jpg"
    wallpaper.touch()

    link_parent = tmp_path / "Images"
    link_parent.mkdir()
    (link_parent / "Wallpapers").symlink_to(tmp_path / "Dropbox" / "Wallpapers")

    cfg = Config(wallpaper_dir=link_parent / "Wallpapers" / "paintings")
    assert cfg.is_art_wallpaper(wallpaper)


def test_nested_subdirectory_matches(tmp_path):
    art = tmp_path / "paintings"
    (art / "impressionists").mkdir(parents=True)
    wallpaper = art / "impressionists" / "monet.jpg"
    wallpaper.touch()

    cfg = Config(wallpaper_dir=art)
    assert cfg.is_art_wallpaper(wallpaper)


def test_paths_are_expanded():
    cfg = Config(wallpaper_dir="~/Images", cache_dir="~/.cache/artwall")
    assert str(cfg.wallpaper_dir).startswith(str(Path.home()))
    assert cfg.state_file.name == "current_art.json"
    assert cfg.cache_file.name == "known_art.json"
