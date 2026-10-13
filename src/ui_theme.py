"""Look and feel for the dashboard: a "traffic control room" theme.

Pure functions that return HTML/CSS strings (no Streamlit import), so they can be
unit-tested. Everything animated is plain CSS: Streamlit strips <script> from
markdown, and CSS also lets us honour prefers-reduced-motion.
"""
import zlib

LEVELS = ["Low", "Moderate", "High", "Severe"]
LEVEL_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Severe": "🔴"}

# One line per mood, picked deterministically per (level, key) so the text does not
# change on every rerun (a message that flickers on each slider drag is irritating).
STATUS_MESSAGES = {
    "Low": ["Smooth sailing. The road is all yours.", "Wide open. Go, go, go!", "Barely a car in sight."],
    "Moderate": ["Busy but moving. Keep your eyes up.", "Steady flow. A little patience.",
                 "Getting lively out there."],
    "High": ["Heavy traffic. Allow extra time.", "Brake lights ahead. Pack some patience.",
             "Slow going. Maybe pick a good podcast."],
    "Severe": ["Standstill alert. Consider another route.", "Gridlock! Time to rethink the trip.",
               "Parking-lot vibes. Avoid if you can."],
}

# Seconds for a car to cross the header strip: flowing traffic is quick, jammed is slow.
HEADER_CROSS_SECONDS = {"Low": 7, "Moderate": 12, "High": 22, "Severe": 40}
DEFAULT_CROSS_SECONDS = 10


def status_message(level: str, key: str = "") -> str:
    """A short friendly line for a congestion level; stable for the same (level, key)."""
    options = STATUS_MESSAGES.get(level)
    if not options:
        return ""
    return options[zlib.crc32(f"{level}|{key}".encode()) % len(options)]


def theme_css() -> str:
    """The <style> block for the whole page."""
    return """
<style>
:root {
  --tc-bg: #070b14;
  --tc-panel: #0d1424;
  --tc-line: #1e2a44;
  --tc-cyan: #22d3ee;
  --tc-text: #e2e8f0;
  --tc-muted: #8aa0c0;
}
.stApp {
  background:
    radial-gradient(1100px 500px at 8% -10%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(168,85,247,.08), transparent 55%),
    var(--tc-bg);
}
div.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
div[data-testid="stMetric"] {
  background: var(--tc-panel); border: 1px solid var(--tc-line); border-radius: 12px; padding: 10px 14px;
}

/* tabs as glowing pills */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 999px; padding: 6px 16px; background: var(--tc-panel); border: 1px solid var(--tc-line);
  transition: box-shadow .2s, border-color .2s;
}
.stTabs [data-baseweb="tab"]:hover { border-color: var(--tc-cyan); }
.stTabs [aria-selected="true"] { border-color: var(--tc-cyan); box-shadow: 0 0 14px rgba(34,211,238,.35); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* buttons */
div[data-testid="stBaseButton-primary"] > button, button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #0891b2, #22d3ee); color: #04121a; font-weight: 700; border: 0;
  box-shadow: 0 0 0 rgba(34,211,238,0); transition: box-shadow .2s, transform .1s;
}
button[data-testid="stBaseButton-primary"]:hover { box-shadow: 0 0 18px rgba(34,211,238,.55); transform: translateY(-1px); }
button[data-testid="stBaseButton-secondary"] { border-color: var(--tc-line); background: var(--tc-panel); }
button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--tc-cyan); }
details { border-color: var(--tc-line) !important; border-radius: 12px !important; }
img { border-radius: 10px; }

/* header */
.tc-header { margin-bottom: 14px; }
.tc-eyebrow { font-size: 12px; letter-spacing: 3px; color: var(--tc-cyan); font-weight: 700; }
.tc-title { font-size: 2.1rem; font-weight: 800; line-height: 1.15; color: var(--tc-text); margin: 2px 0 4px; }
.tc-live { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700;
  letter-spacing: 2px; color: #fca5a5; margin-left: 12px; vertical-align: middle; }
.tc-live i { width: 9px; height: 9px; border-radius: 50%; background: #ef4444; display: inline-block;
  box-shadow: 0 0 8px #ef4444; animation: tc-blink 1.4s ease-in-out infinite; }
.tc-sub { color: var(--tc-muted); font-size: 14px; margin-bottom: 10px; }
.tc-road { position: relative; height: 34px; overflow: hidden; border-radius: 8px; background: #0b1220;
  border: 1px solid var(--tc-line);
  background-image: repeating-linear-gradient(90deg, rgba(148,163,184,.35) 0 18px, transparent 18px 42px);
  background-size: 100% 2px; background-repeat: no-repeat; background-position: 0 50%; }
.tc-car { position: absolute; top: 3px; left: -40px; font-size: 22px; display: inline-block;
  transform: scaleX(-1); animation: tc-drive var(--tc-dur, 10s) linear infinite; }
@keyframes tc-drive { from { left: -40px; } to { left: calc(100% + 20px); } }
@keyframes tc-blink { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }

/* congestion badge */
.tc-badge { border-radius: 16px; text-align: center; margin-bottom: 10px; }
.tc-badge .tc-badge-title { font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.tc-badge .tc-badge-level { font-weight: 800; line-height: 1.2; }
.tc-badge .tc-badge-msg { margin-top: 4px; color: var(--tc-text); opacity: .85; }
.tc-pulse { animation: tc-glow 1.8s ease-in-out infinite; }
@keyframes tc-glow { 0%, 100% { box-shadow: 0 0 8px var(--tc-c); } 50% { box-shadow: 0 0 26px var(--tc-c); } }

/* animated counter: --tc-n counts from 0 to the value set inline */
@property --tc-n { syntax: '<integer>'; initial-value: 0; inherits: false; }
.tc-counter { background: var(--tc-panel); border: 1px solid var(--tc-line); border-radius: 12px;
  padding: 10px 14px; }
.tc-counter .tc-counter-label { font-size: 14px; color: var(--tc-muted); }
.tc-counter .tc-counter-num { font-size: 2.2rem; font-weight: 800; color: var(--tc-cyan);
  text-shadow: 0 0 14px rgba(34,211,238,.45); counter-reset: tcn var(--tc-n);
  animation: tc-count .9s ease-out; }
.tc-counter .tc-counter-num::after { content: counter(tcn); }
.tc-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
@keyframes tc-count { from { --tc-n: 0; } }

@media (prefers-reduced-motion: reduce) {
  .tc-car, .tc-live i, .tc-pulse, .tc-counter .tc-counter-num { animation: none !important; }
}
</style>
"""


def header_html(title: str, subtitle: str, level: str | None = None) -> str:
    """The page header: eyebrow, title, LIVE dot, and a strip of cars whose speed follows
    the latest congestion level (jammed roads make them crawl)."""
    seconds = HEADER_CROSS_SECONDS.get(level, DEFAULT_CROSS_SECONDS)
    cars = ["🚗", "🚕", "🚙", "🚌", "🚓"]
    car_html = "".join(
        f'<span class="tc-car" style="animation-delay:-{i * seconds / len(cars):.2f}s">{c}</span>'
        for i, c in enumerate(cars)
    )
    return (f'<div class="tc-header"><div class="tc-eyebrow">TRAFFIC CONTROL ROOM</div>'
            f'<div class="tc-title">🚦 {title}<span class="tc-live"><i></i>LIVE</span></div>'
            f'<div class="tc-sub">{subtitle}</div>'
            f'<div class="tc-road" style="--tc-dur:{seconds}s">{car_html}</div></div>')


def badge_html(level: str, color: str, small: bool = False, message: str = "") -> str:
    """The congestion badge: a glowing card; High and Severe pulse."""
    pad, title_size, level_size = ("10px 14px", "11px", "22px") if small else ("18px 24px", "13px", "44px")
    pulse = " tc-pulse" if level in ("High", "Severe") else ""
    msg = (f'<div class="tc-badge-msg" style="font-size:{"12px" if small else "15px"}">{message}</div>'
           if message else "")
    return (f'<div class="tc-badge{pulse}" style="--tc-c:{color};background:{color}1a;'
            f'border:2px solid {color};padding:{pad};box-shadow:0 0 12px {color}55;">'
            f'<div class="tc-badge-title" style="font-size:{title_size};color:{color}">Congestion</div>'
            f'<div class="tc-badge-level" style="font-size:{level_size};color:{color}">'
            f'{LEVEL_EMOJI.get(level, "")} {level}</div>{msg}</div>')


def counter_html(value: int, label: str) -> str:
    """A number that counts up from 0 when it first appears (pure CSS). The animated digits are
    drawn by CSS, so the real value is also in a visually hidden span for screen readers and copy."""
    value = max(0, int(value))
    return (f'<div class="tc-counter"><div class="tc-counter-label">{label}</div>'
            f'<div class="tc-counter-num" style="--tc-n:{value}" aria-hidden="true"></div>'
            f'<span class="tc-sr">{value}</span></div>')
