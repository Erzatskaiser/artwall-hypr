"""How a CLIP score turns into what the overlay shows."""

import json

import pytest

pytest.importorskip("torch", reason="artwall.intelligence imports torch at module level")

from artwall.config import Config  # noqa: E402
from artwall.intelligence import ArtIntelligence  # noqa: E402


class FakeIndex:
    """Stands in for the CLIP index so these tests need no model and no embeddings."""

    def __init__(self, score, meta=None):
        self.score = score
        self.meta = meta or {"title": "Starry Night", "artist": "Van Gogh", "year": "1889"}

    def search(self, image_path):
        return self.meta, self.score


def make(tmp_path, score, **cfg_kwargs):
    cfg = Config(wallpaper_dir=tmp_path, cache_dir=tmp_path / "cache", **cfg_kwargs)
    intel = ArtIntelligence(cfg)
    intel.clip_index = FakeIndex(score)
    wallpaper = tmp_path / "wall.jpg"
    wallpaper.write_bytes(b"only the hash of these bytes matters here")
    return cfg, intel, wallpaper


def test_confident_match_names_the_artwork(tmp_path):
    cfg, intel, wallpaper = make(tmp_path, 0.95)
    meta = intel.process_wallpaper(wallpaper)
    assert meta["title"] == "Starry Night"
    assert meta["artist"] == "Van Gogh"
    assert meta["year"] == "1889"


def test_fuzzy_match_is_reported_as_a_resemblance(tmp_path):
    cfg, intel, wallpaper = make(tmp_path, 0.80)
    meta = intel.process_wallpaper(wallpaper)
    assert meta["title"] == "Reminiscent of: Starry Night"


def test_weak_match_can_be_suppressed(tmp_path):
    cfg, intel, wallpaper = make(tmp_path, 0.30, show_weak_matches=False)
    meta = intel.process_wallpaper(wallpaper)
    assert meta["title"] == "Unknown Art"


def test_weak_match_shown_when_enabled(tmp_path):
    cfg, intel, wallpaper = make(tmp_path, 0.30, show_weak_matches=True)
    meta = intel.process_wallpaper(wallpaper)
    assert meta["title"].startswith("Reminiscent of:")


def test_result_is_cached_and_reused(tmp_path):
    cfg, intel, wallpaper = make(tmp_path, 0.95)
    intel.process_wallpaper(wallpaper)

    cached = json.loads(cfg.cache_file.read_text())
    assert list(cached.values())[0]["title"] == "Starry Night"

    # A second pass must not need the index at all
    intel.clip_index = None
    assert intel.process_wallpaper(wallpaper)["title"] == "Starry Night"


def test_old_cache_entries_lose_their_generated_description(tmp_path):
    cfg = Config(wallpaper_dir=tmp_path, cache_dir=tmp_path / "cache")
    cfg.cache_dir.mkdir(parents=True)
    cfg.cache_file.write_text(
        json.dumps({"abc": {"title": "T", "artist": "A", "year": "", "description": "old text"}})
    )

    intel = ArtIntelligence(cfg)
    assert "description" not in intel.known_art["abc"]
