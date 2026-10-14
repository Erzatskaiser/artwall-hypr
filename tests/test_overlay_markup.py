import json

import pytest

gi = pytest.importorskip("gi", reason="PyGObject and GTK are needed for the overlay")
gi.require_version("Gtk", "3.0")

from artwall.overlay import build_markup, load_metadata  # noqa: E402


def test_title_artist_and_year():
    markup = build_markup(
        {"title": "Starry Night", "artist": "Van Gogh", "year": "1889", "description": "Swirling."}
    )
    assert "<b>Starry Night</b> - Van Gogh, 1889" in markup
    assert markup.endswith("Swirling.")


def test_ampersands_are_escaped():
    """
    Pango rejects the whole string on an unescaped &, which left the overlay blank for
    any artwork with one in its title.
    """
    markup = build_markup({"title": "Arts & Crafts", "artist": "Morris & Co", "description": ""})
    assert "&amp;" in markup
    assert "Arts & Crafts" not in markup


def test_angle_brackets_are_escaped():
    markup = build_markup({"title": "<script>", "artist": "", "description": ""})
    assert "&lt;script&gt;" in markup


def test_blank_description_lines_are_dropped():
    markup = build_markup(
        {"title": "T", "artist": "A", "description": "first\n\n   \nsecond\n"}
    )
    assert markup.count("\n") == 2
    assert markup.endswith("first\nsecond")


def test_missing_fields_fall_back():
    markup = build_markup({})
    assert "Unknown" in markup


def test_load_metadata_handles_missing_file(tmp_path):
    meta = load_metadata(tmp_path / "nope.json")
    assert meta["title"]


def test_load_metadata_handles_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    meta = load_metadata(bad)
    assert meta["title"]


def test_load_metadata_reads_valid_file(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"title": "T", "artist": "A", "year": "", "description": "D"}))
    assert load_metadata(good)["title"] == "T"
