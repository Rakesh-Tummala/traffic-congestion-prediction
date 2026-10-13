"""Tests for the live-video embed helper (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from live_video import HLS_JS_URL, hls_player_html, is_playable_url  # noqa: E402

GOOD = "https://wzmedia.dot.ca.gov/D5/101atRoute41.stream/playlist.m3u8"


def test_is_playable_url_accepts_hls_playlists_only():
    assert is_playable_url(GOOD)
    assert not is_playable_url("")
    assert not is_playable_url(None)
    assert not is_playable_url("javascript:alert(1)//x.m3u8")
    assert not is_playable_url("ftp://example.com/a.m3u8")
    assert not is_playable_url("https://example.com/picture.jpg")


def test_player_html_embeds_the_stream_and_the_pinned_hls_library():
    html = hls_player_html(GOOD)
    assert GOOD in html and HLS_JS_URL in html and "hls.js/1.5.13" in HLS_JS_URL
    assert "isn't responding" in html  # dead streams get a message, not a blank box


def test_player_html_quotes_the_url_safely():
    tricky = 'https://x.example.com/a";alert(1);//.m3u8'
    html = hls_player_html(tricky)
    assert 'a";alert(1)' not in html          # the raw quote must not break out of the JS string
    assert '\\"' in html


def test_player_rejects_bad_urls():
    with pytest.raises(ValueError):
        hls_player_html("https://example.com/not-a-stream.jpg")
