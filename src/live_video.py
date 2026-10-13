"""Embed a Caltrans live video stream (HLS) in the dashboard.

Caltrans publishes an HLS playlist (.m3u8) for most cameras (`stream_url` in
live_cameras.fetch_cameras). The video plays in the viewer's own browser, straight
from Caltrans's server (their stream answers with `Access-Control-Allow-Origin: *`),
using hls.js; Safari plays HLS natively. Nothing is proxied or recorded here.
Roughly a third of the streams sampled while building this were dead, so the player
shows a message instead of a blank box when the stream does not respond.
"""
import json
from urllib.parse import urlparse

HLS_JS_URL = "https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.13/hls.min.js"


def is_playable_url(url: str) -> bool:
    """Only http(s) URLs that look like an HLS playlist are embedded."""
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc) and parsed.path.endswith(".m3u8")


def hls_player_html(stream_url: str, height: int = 360) -> str:
    """A self-contained HTML page with a muted, autoplaying live-video element."""
    if not is_playable_url(stream_url):
        raise ValueError(f"Not a playable HLS stream URL: {stream_url!r}")
    url_js = json.dumps(stream_url)  # safely quoted for use inside the script
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#070b14;color:#cbd5e1;font-family:sans-serif}}
#wrap{{position:relative;height:{height}px;border-radius:12px;overflow:hidden;border:1px solid #1e2a44}}
video{{width:100%;height:100%;background:#000;object-fit:contain}}
#live{{position:absolute;top:10px;left:12px;font:700 11px sans-serif;letter-spacing:2px;color:#fca5a5;
background:rgba(0,0,0,.55);padding:3px 8px;border-radius:999px}}
#live i{{display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:6px}}
#msg{{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;
padding:20px;font-size:14px;background:rgba(7,11,20,.88)}}
</style></head><body><div id="wrap">
<video id="v" muted autoplay playsinline controls></video>
<div id="live"><i></i>LIVE</div>
<div id="msg">This camera's video stream isn't responding right now. Many Caltrans streams are
offline at any given time, so try another camera (snapshots still work).</div></div>
<script src="{HLS_JS_URL}"></script>
<script>
const url = {url_js}, v = document.getElementById('v'), msg = document.getElementById('msg');
function fail(){{ msg.style.display = 'flex'; }}
if (window.Hls && Hls.isSupported()) {{
  const hls = new Hls({{ lowLatencyMode: true }});
  hls.loadSource(url); hls.attachMedia(v);
  hls.on(Hls.Events.ERROR, (e, d) => {{ if (d.fatal) fail(); }});
  setTimeout(() => {{ if (v.readyState < 2) fail(); }}, 15000);
}} else if (v.canPlayType('application/vnd.apple.mpegurl')) {{
  v.src = url; v.addEventListener('error', fail);
}} else {{ fail(); }}
</script></body></html>"""
