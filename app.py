"""Streamlit dashboard: pick a real live Caltrans camera (click the map or use
the dropdown — they stay in sync), upload an image, or paste a snapshot URL;
set conditions; get vehicle detection plus a fused congestion prediction from
all three trained models, with an anomaly flag, a feature-level explanation,
a per-camera history trend, and a "when should I leave" recommendation. Also
includes a monitoring grid to check several cameras at once, and a route tab
to combine several cameras into one worst-segment-wins reading. Run with:
streamlit run app.py"""
import os
import sys
import time
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor

import altair as alt
import cv2
import folium
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cv_module import analyze_frame, congestion_from_density, load_frame  # noqa: E402
from camera_time import camera_now  # noqa: E402
from ui_theme import badge_html, counter_html, header_html, status_message, theme_css  # noqa: E402
from data_loader import WEATHER_CODES  # noqa: E402
from live_cameras import DISTRICTS, fetch_cameras  # noqa: E402
from live_predict import (  # noqa: E402
    CONGESTION_LEVELS, MODELS_DIR, build_feature_row, calibrated_label, congestion_bins,
    detect_anomaly, forecast_day, predict_all_models, recommend_departure,
)
from speed_estimation import US_LANE_WIDTH_M, annotate_speeds, check_speeds  # noqa: E402
from echallan import (  # noqa: E402
    DISCLAIMER as ECHALLAN_DISCLAIMER, FINE_SCHEDULE_INR, append_to_log, generate_challan, load_log,
    render_challan_html, clear_log,
)
from live_monitor import LiveMonitor  # noqa: E402
from speed_limits import lookup_async  # noqa: E402
from live_video import hls_player_html, is_playable_url  # noqa: E402
from anpr import read_plate  # noqa: E402
from vehicle_attributes import describe_vehicle  # noqa: E402
from prediction_log import load_history, log_prediction  # noqa: E402
from explain import explain_prediction  # noqa: E402
from weather import fetch_current_weather  # noqa: E402
from report import build_report_png  # noqa: E402
from route_map import (  # noqa: E402
    CAMERA_ON_ROUTE_M, FAR_CAMERA_DISTANCE_M, MAX_CAMERA_DISTANCE_M, cameras_near_route, color_route,
    fetch_route, fit_view, geocode, order_cameras,
)

LEVEL_COLORS = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
LEVEL_COLORS_RGB = {"Low": [34, 197, 94], "Moderate": [234, 179, 8], "High": [249, 115, 22], "Severe": [239, 68, 68]}
MODEL_LABELS = {"linear_regression": "Linear Regression", "svr": "SVR", "lstm": "LSTM"}
MODEL_R2 = {"linear_regression": 0.948, "svr": 0.960, "lstm": 0.977}
# Held-out test-set MAE (mean absolute error, in vehicles/hour) for each trained
# model — computed once from the saved model artifacts, used only to draw an
# honest "predicted ± typical error" band on the forecast chart rather than
# implying false precision with a single confident line.
MODEL_MAE = {"linear_regression": 334.2, "svr": 253.5, "lstm": 202.6}
MODELS_READY = all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ("linear_regression.joblib", "svr.joblib", "lstm.pt", "lstm_scalers.joblib", "seasonal_profile.joblib")
)


@st.cache_data(ttl=300)
def _cached_cameras(district: int):
    return fetch_cameras(district)


@st.cache_data(ttl=1800)
def _cached_forecast(date_str: str, temp_c: float, rain_1h: float, snow_1h: float,
                      clouds_all: float, is_holiday: int, weather_main: str, model: str):
    base = datetime.strptime(date_str, "%Y-%m-%d")
    return forecast_day(base, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main, model)


@st.cache_data(ttl=1800, show_spinner=False)
def _all_cameras():
    """Every Caltrans district's cameras in one list, so a route between any two
    places can find the cameras along it. Districts that fail to load are skipped
    rather than failing the whole route."""
    def one(district):
        try:
            return fetch_cameras(district)
        except Exception:
            return []
    with ThreadPoolExecutor(max_workers=6) as pool:
        return [cam for batch in pool.map(one, DISTRICTS) for cam in batch]


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_route(start: tuple, end: tuple):
    # One OSRM call per distinct start/end pair; failures raise and aren't cached.
    return fetch_route(start, end)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_analyze_frame(frame: np.ndarray):
    # Keyed on the frame's own contents (Streamlit hashes ndarrays), so
    # unrelated reruns — dragging a weather slider, an auto-refresh poll
    # tick that isn't due yet — reuse the same detection instead of paying
    # for YOLO inference again on a frame that hasn't actually changed.
    return analyze_frame(frame)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_predict_all(now_minute: str, temp_c: float, rain_1h: float, snow_1h: float,
                         clouds_all: float, is_holiday: int, weather_main: str,
                         lag_1h: float, lag_3h: float, lag_24h: float):
    now = datetime.strptime(now_minute, "%Y-%m-%d %H:%M")
    return predict_all_models(now, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main,
                               lag_1h=lag_1h, lag_3h=lag_3h, lag_24h=lag_24h)


def _limit_for_camera(cam: dict) -> dict | None:
    """Posted speed limit for a camera's road from OpenStreetMap, without ever blocking the page.
    While the lookup runs in the background this returns the provisional fallback (if any) and sets
    st.session_state["_limit_pending"] so the page can poll and refresh when the real value arrives."""
    try:
        lat, lon = float(cam["latitude"]), float(cam["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    state = lookup_async(lat, lon, cam.get("route", ""))
    st.session_state["_limit_pending"] = state["state"] == "pending"
    return state["result"]


@st.fragment(run_every=2.0)
def _limit_poller():
    """Runs only while a lookup is pending: when it finishes, do one full rerun to apply the result."""
    if not st.session_state.get("_limit_pending"):
        return
    st.caption("🛣️ Looking up this road's speed limit on OpenStreetMap (a public server that is sometimes "
               "slow). The page updates by itself when it answers.")
    cam = st.session_state.get("_limit_cam_for_poll")
    if cam and lookup_async(float(cam["latitude"]), float(cam["longitude"]), cam.get("route", ""))["state"] == "done":
        st.rerun()


def _apply_auto_limit(cam_key: str, info: dict | None, limit_key: str, ref_key: str | None = None) -> None:
    """When the selected camera changes, set the limit widget (and typical-speed widget) from `info`.
    Done only on a camera change so a value the user types afterwards is never overwritten."""
    marker = (cam_key, bool(info and info.get("provisional")))  # re-apply once a provisional value is replaced
    if st.session_state.get(f"_auto_limit_cam_{limit_key}") == marker:
        return
    if info and info["limit_mph"]:
        st.session_state[limit_key] = float(info["limit_mph"])
        if ref_key:
            st.session_state[ref_key] = float(max(10, round(info["limit_mph"] * 0.9 / 5) * 5))
    st.session_state[f"_auto_limit_cam_{limit_key}"] = marker


def _limit_caption(info: dict | None, current: float) -> tuple[str, str]:
    """(caption text, limit_source text for the challan) for the limit currently in the box."""
    if info and info["limit_mph"]:
        if abs(current - info["limit_mph"]) < 0.5:
            if info.get("provisional"):
                return (f"🛣️ {info['detail']}",
                        f"assumed {info['limit_mph']:.0f} mph: OpenStreetMap was unreachable, so the usual "
                        "California freeway limit was used, not a posted sign")
            if info["source"] == "assumed":
                return f"🛣️ {info['detail']}", f"assumed from OpenStreetMap road class: {info['detail']}"
            return f"🛣️ {info['detail']}", info["detail"]
        return (f"🛣️ OpenStreetMap suggested {info['limit_mph']:.0f} mph ({info['detail']}) but you changed it.",
                f"entered manually (OpenStreetMap suggested {info['limit_mph']:.0f} mph)")
    return ("🛣️ Couldn't find this road's limit on OpenStreetMap right now, so it's a manual value.",
            "entered manually")


def congestion_badge(level: str, small: bool = False, key: str = ""):
    """Glowing congestion card with a friendly one-liner (omitted on the small grid cards)."""
    if not small:
        st.session_state["_last_level"] = level  # drives the speed of the cars in the page header
    st.markdown(
        badge_html(level, LEVEL_COLORS[level], small=small,
                   message="" if small else status_message(level, key)),
        unsafe_allow_html=True,
    )


def forecast_chart(df: pd.DataFrame, current_hour: int, bins: np.ndarray, mae: float = None) -> alt.Chart:
    bands = pd.DataFrame({
        "y0": bins[:-1], "y1": bins[1:], "label": CONGESTION_LEVELS,
    })
    band_chart = alt.Chart(bands).mark_rect(opacity=0.12).encode(
        y="y0:Q", y2="y1:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=CONGESTION_LEVELS, range=list(LEVEL_COLORS.values())),
                         legend=alt.Legend(title="Congestion level")),
    )
    chart = band_chart

    if mae is not None:
        # A single confident-looking line overstates precision — shading
        # ± the model's own held-out test-set MAE gives an honest sense of
        # typical error instead of implying the forecast is exact.
        error_df = df.copy()
        error_df["low"] = (error_df["volume"] - mae).clip(lower=0)
        error_df["high"] = error_df["volume"] + mae
        error_band = alt.Chart(error_df).mark_area(opacity=0.18, color="#38bdf8").encode(
            x="hour:Q", y="low:Q", y2="high:Q",
        )
        chart = chart + error_band

    line = alt.Chart(df).mark_line(color="#e5e7eb", strokeWidth=2.5).encode(
        x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])),
        y=alt.Y("volume:Q", title="Predicted traffic volume"),
    )
    now_point = alt.Chart(df[df["hour"] == current_hour]).mark_point(
        size=160, color="white", filled=True, stroke="black", strokeWidth=2,
    ).encode(x="hour:Q", y="volume:Q", tooltip=["hour", "volume", "label"])
    return (chart + line + now_point).properties(height=280).interactive()


def _is_valid_index(value, n: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < n


def _hex_to_rgb(hex_color: str) -> list:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]


def _last_known_labels() -> dict:
    """camera name -> most recently logged final congestion label, from this
    install's local prediction history (empty if a camera has never been
    checked). History is oldest-first, so later entries simply overwrite
    earlier ones for the same camera, leaving the latest."""
    latest = {}
    for entry in load_history(limit=3000):
        latest[entry["camera_name"]] = entry["final_label"]
    return latest


def camera_picker_map(filtered: list, selected_idx_key: str):
    """Render a clickable map of `filtered` cameras, with each point's color
    showing its last-known congestion level (gray if never checked yet).
    Clicking a point selects that camera by writing into
    st.session_state[selected_idx_key] — the same session-state key the
    paired selectbox is bound to — so map clicks and dropdown choices stay
    in sync in either direction."""
    if selected_idx_key not in st.session_state:
        st.session_state[selected_idx_key] = 0
    # A selectbox remembers its choice as the option's *label*. When a filter or
    # district change drops that label from the new options, Streamlit hands the
    # old label string back instead of an index, so this value can be a str, not
    # just an out-of-range int (found via live testing: TypeError on a bare
    # `>= len(filtered)`).
    if not _is_valid_index(st.session_state[selected_idx_key], len(filtered)):
        st.session_state[selected_idx_key] = 0
    current_idx = st.session_state[selected_idx_key]

    # Built as plain Python int/float/str (not numpy scalars): pydeck/deck.gl's
    # JSON encoding of numpy-dtype DataFrame columns silently breaks the
    # component's picking (points render, but clicks never register a
    # selection) — confirmed by isolating this during development.
    rows = [
        {"lat": float(c["latitude"]), "lon": float(c["longitude"]), "name": str(c["name"]), "cam_idx": int(i)}
        for i, c in enumerate(filtered) if c["latitude"] and c["longitude"]
    ]
    map_df = pd.DataFrame(rows, dtype=object)
    if map_df.empty:
        return

    # A separate, non-pickable layer carries the per-row congestion colors.
    # Per-row accessors are exactly what breaks click-picking on the *pickable*
    # layer below (confirmed during earlier development) — but that finding is
    # specific to picking, not rendering, so per-row colors are safe here since
    # this layer never needs to be clicked.
    latest_labels = _last_known_labels()
    unknown_rgba = [100, 116, 139, 60]
    congestion_rows = [
        {"lat": r["lat"], "lon": r["lon"],
         "color": (_hex_to_rgb(LEVEL_COLORS[latest_labels[r["name"]]]) + [200]
                   if r["name"] in latest_labels else unknown_rgba)}
        for r in rows
    ]
    congestion_layer = pdk.Layer(
        "ScatterplotLayer", id="congestion-colors", data=pd.DataFrame(congestion_rows, dtype=object),
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=350,
        pickable=False,
    )

    # The pickable layer uses ONLY static (non-per-row) color/radius — see the
    # note above. It sits on top of the congestion-color halo as a small,
    # consistent click target.
    base_layer = pdk.Layer(
        "ScatterplotLayer", id="cameras", data=map_df,
        get_position=["lon", "lat"],
        get_color="[100, 116, 139, 160]",
        get_radius=200,
        pickable=True,
        auto_highlight=True,
    )
    selected_row = map_df[map_df["cam_idx"] == current_idx]
    # A hollow ring, not a filled dot: a filled highlight sits on top of the
    # congestion-color halo and fully hides it for exactly the camera the
    # user is looking at (found via live testing).
    highlight_layer = pdk.Layer(
        "ScatterplotLayer", id="selected-camera", data=selected_row,
        get_position=["lon", "lat"],
        stroked=True, filled=False,
        get_line_color="[255, 255, 255, 255]",
        get_radius=600,
        line_width_min_pixels=3,
        pickable=False,
    )
    view = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=8, controller=True)
    map_state = st.pydeck_chart(
        pdk.Deck(layers=[congestion_layer, base_layer, highlight_layer], initial_view_state=view,
                 map_style=None, tooltip={"text": "{name}"}),
        on_select="rerun", selection_mode="single-object", key=f"{selected_idx_key}_map",
    )
    st.caption("Click a point on the map to select that camera, or use the dropdown below. "
              "Color shows each camera's last-known congestion level on this install "
              "(gray = not checked yet).")

    selected_objs = map_state.selection.objects.get("cameras", []) if map_state else []
    if selected_objs:
        clicked_idx = int(selected_objs[0]["cam_idx"])
        last_click_marker = f"_{selected_idx_key}_last_click"
        if st.session_state.get(last_click_marker) != clicked_idx:
            st.session_state[last_click_marker] = clicked_idx
            st.session_state[selected_idx_key] = clicked_idx


UNKNOWN_RGBA = [120, 130, 140, 170]


def _route_rgba(label, alpha: int) -> list:
    return LEVEL_COLORS_RGB[label] + [alpha] if label else UNKNOWN_RGBA


def _route_map(ordered: list, route: dict = None, endpoints: tuple = None):
    """Draw a route: the real road path colored stretch by stretch from the nearest
    camera's reading (solid within 2 km, faded up to 8 km, gray beyond), every
    camera as a numbered dot, and A/B markers for the endpoints when the route
    came from places or map clicks. `ordered` is route results in route order.
    When no road path is given it is fetched between the first and last camera,
    falling back to straight lines if that fails. Nothing here is clickable, so
    per-row colors are safe (see camera_picker_map)."""
    mapped = [r for r in ordered if r["camera"].get("latitude") and r["camera"].get("longitude")]
    if route is None and len(mapped) < 2:
        return

    def latlon(r):
        return float(r["camera"]["latitude"]), float(r["camera"]["longitude"])

    readings = [(*latlon(r), r["final_label"]) for r in mapped if r["final_label"]]
    route_error = None
    if route is None:
        try:
            route = _cached_route(latlon(mapped[0]), latlon(mapped[-1]))
        except Exception as e:
            route_error = str(e)

    if route:
        runs = color_route(route["points"], readings)
    else:
        runs = []
        for a, b in zip(mapped, mapped[1:]):
            labels = [x["final_label"] for x in (a, b) if x["final_label"]]
            (lat_a, lon_a), (lat_b, lon_b) = latlon(a), latlon(b)
            runs.append({"label": max(labels, key=CONGESTION_LEVELS.index) if labels else None, "near": True,
                         "path": [[lon_a, lat_a], [lon_b, lat_b]]})

    path_rows = [{"path": run["path"], "color": _route_rgba(run["label"], 235 if run["near"] else 120)}
                 for run in runs]
    camera_rows = [{"lon": latlon(r)[1], "lat": latlon(r)[0], "n": str(i + 1),
                    "color": _route_rgba(r["final_label"], 255)} for i, r in enumerate(mapped)]
    end_rows = ([{"lon": lon, "lat": lat, "n": letter} for letter, (lat, lon) in zip("AB", endpoints)]
                if endpoints else [])
    all_points = ([pt for run in runs for pt in run["path"]] + [[c["lon"], c["lat"]] for c in camera_rows]
                  + [[e["lon"], e["lat"]] for e in end_rows])

    layers = [
        pdk.Layer("PathLayer", id="route-path", data=path_rows, get_path="path", get_color="color",
                  width_min_pixels=6, cap_rounded=True, joint_rounded=True),
        pdk.Layer("ScatterplotLayer", id="route-cameras", data=camera_rows, get_position=["lon", "lat"],
                  get_fill_color="color", get_line_color=[255, 255, 255, 255], stroked=True,
                  line_width_min_pixels=2, radius_min_pixels=12, get_radius=60),
        pdk.Layer("TextLayer", id="route-numbers", data=camera_rows, get_position=["lon", "lat"],
                  get_text="n", get_size=14, get_color=[10, 10, 10, 255],
                  get_text_anchor="'middle'", get_alignment_baseline="'center'"),
    ]
    if end_rows:
        layers += [
            pdk.Layer("ScatterplotLayer", id="route-ends", data=end_rows, get_position=["lon", "lat"],
                      get_fill_color=[255, 255, 255, 255], get_line_color=[10, 10, 10, 255], stroked=True,
                      line_width_min_pixels=3, radius_min_pixels=14, get_radius=80),
            pdk.Layer("TextLayer", id="route-end-letters", data=end_rows, get_position=["lon", "lat"],
                      get_text="n", get_size=16, get_color=[10, 10, 10, 255], font_weight=700,
                      get_text_anchor="'middle'", get_alignment_baseline="'center'"),
        ]
    if not all_points:
        return
    lat, lon, zoom = fit_view(all_points)
    st.pydeck_chart(
        pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                 map_style=None),
        height=420,
    )

    if route:
        st.caption(f"Road path from OpenStreetMap (OSRM): {route['distance_m'] / 1000:.1f} km, about "
                   f"{route['duration_s'] / 60:.0f} min without traffic.")
    else:
        st.caption(f"Couldn't fetch the road path ({route_error}); showing straight lines between cameras instead.")
    st.caption(f"Colors come from the nearest camera: solid within {MAX_CAMERA_DISTANCE_M / 1000:.0f} km, "
               f"faded {MAX_CAMERA_DISTANCE_M / 1000:.0f}–{FAR_CAMERA_DISTANCE_M / 1000:.0f} km away "
               f"(a weaker guess), gray = no camera within {FAR_CAMERA_DISTANCE_M / 1000:.0f} km. These are "
               "estimates from scattered camera snapshots, not continuous traffic data like a navigation app "
               "has. Numbers match the list below.")


st.set_page_config(page_title="Smart Traffic Congestion Prediction", layout="wide", page_icon="🚦")
st.markdown(theme_css(), unsafe_allow_html=True)
st.markdown(
    header_html("Smart Traffic Congestion Prediction",
                "Computer vision vehicle counting + Linear Regression / SVR / LSTM, fused with real live "
                "traffic cameras. The cars above slow down when the latest reading is congested.",
                st.session_state.get("_last_level")),
    unsafe_allow_html=True,
)

mode_single, mode_grid, mode_route, mode_challan = st.tabs(
    ["🔍 Single camera", "🗂️ Monitoring grid", "🛣️ Route", "🧾 Live e-Challan"])
mode_live = mode_challan  # the live monitor and the e-challan section share one tab, live part first

# ---------------------------------------------------------------- single camera
with mode_single:
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("1. Camera frame")
        tab_live, tab_upload, tab_url = st.tabs(["Live Caltrans camera", "Upload image", "Snapshot URL"])
        # Streamlit renders every tab's body on every rerun regardless of which
        # tab is visually active — a tab's *widgets* keep their last value
        # (leftover text in the URL box, a still-attached upload) even while
        # the user is interacting with a different tab. Without explicit
        # "which source acted most recently" tracking, whichever tab's block
        # happens to run last in the script (URL, here) would silently
        # re-fetch its stale input and overwrite whatever the user just did
        # in another tab — confirmed happening during testing: leftover text
        # in the Snapshot URL box kept overriding a fresh Live-camera fetch on
        # every single rerun. `frame_source` is only updated on a genuinely
        # new action (button click / new upload / changed URL text), and only
        # that source's frame is used below.
        st.session_state.setdefault("frame_source", None)
        frame = None
        live_stream_url = None  # only set when the current frame came from a live Caltrans camera
        live_cam_name = None

        with tab_live:
            st.caption("Real, currently in-service Caltrans traffic cameras — "
                       "no API key needed. Public data: cwwp2.dot.ca.gov")
            district = st.selectbox("District", DISTRICTS, index=6,
                                     format_func=lambda d: f"District {d}")
            try:
                cameras = _cached_cameras(district)
            except Exception as e:
                cameras = []
                st.error(f"Could not reach Caltrans camera feed: {e}")

            if cameras:
                search = st.text_input("Filter by route / location", "")
                filtered = cameras
                if search:
                    needle = search.lower()
                    filtered = [c for c in cameras if needle in c["name"].lower() or needle in c["nearby_place"].lower()]

                if filtered:
                    camera_picker_map(filtered, "camera_idx")

                    labels = [f"{c['name']} — {c['nearby_place']}" for c in filtered]
                    choice = st.selectbox(f"Camera ({len(filtered)} available)", range(len(filtered)),
                                          format_func=lambda i: labels[i], key="camera_idx")
                    if not isinstance(choice, int):  # stale label after a filter change; see camera_picker_map
                        st.rerun()
                    cam = filtered[choice]

                    fetch_col, weather_col, refresh_col = st.columns([1.3, 1.3, 1])
                    with fetch_col:
                        if st.button("Fetch live snapshot", type="primary"):
                            try:
                                st.session_state["live_frame"] = load_frame(cam["image_url"])
                                st.session_state["live_frame_stream_url"] = cam.get("stream_url", "")
                                st.session_state["live_frame_cam_name"] = cam["name"]
                                st.session_state["live_frame_cam"] = cam
                                st.session_state["live_frame_cam_lat"] = cam.get("latitude")
                                st.session_state["live_frame_cam_lon"] = cam.get("longitude")
                                st.session_state["frame_source"] = "live"
                                st.session_state["_last_auto_fetch_ts"] = time.time()
                            except Exception as e:
                                st.error(f"Could not load frame: {e}")

                    with weather_col:
                        if st.button("🌤️ Use real weather here", disabled=not (cam.get("latitude") and cam.get("longitude"))):
                            try:
                                real_weather = fetch_current_weather(cam["latitude"], cam["longitude"])
                                # Clamped to each widget's declared range, and cast to match its
                                # declared type (the cloud-cover slider is int-typed) — Streamlit
                                # raises if a pre-set session_state value falls outside a
                                # slider's bounds or mixes int/float with its declared type.
                                st.session_state["weather_temp"] = float(np.clip(real_weather["temp_c"], -20.0, 45.0))
                                st.session_state["weather_rain"] = float(np.clip(real_weather["rain_1h"], 0.0, 100.0))
                                st.session_state["weather_snow"] = float(np.clip(real_weather["snow_1h"], 0.0, 100.0))
                                st.session_state["weather_clouds"] = int(np.clip(round(real_weather["clouds_all"]), 0, 100))
                                st.session_state["weather_main_select"] = real_weather["weather_main"]
                                st.session_state["_real_weather_fetched"] = True
                            except Exception as e:
                                st.error(f"Could not fetch real weather: {e}")

                    with refresh_col:
                        auto_refresh = st.checkbox("🔴 Auto-refresh", key="auto_refresh_enabled",
                                                   help="Keep re-fetching this camera automatically.")

                    if st.session_state.get("_real_weather_fetched"):
                        st.caption("✅ Weather conditions below were auto-filled from this camera's real "
                                  "location via Open-Meteo (free, no API key) — edit them if you'd rather "
                                  "set your own.")

                    if auto_refresh:
                        refresh_secs = st.slider("Refresh every (seconds)", 10, 120, 30, key="auto_refresh_secs")
                    else:
                        st.session_state.pop("_last_auto_fetch_ts", None)

                    if is_playable_url(cam.get("stream_url", "")):
                        if st.checkbox("📹 Watch live video", key="watch_live_video",
                                       help="Plays this camera's live stream straight from Caltrans in your "
                                            "browser. Detection and prediction still use the snapshot."):
                            components.html(hls_player_html(cam["stream_url"]), height=372)
                            st.caption("Live video plays directly from Caltrans in your browser; many Caltrans "
                                       "streams are offline at any time. The congestion reading is still computed "
                                       "from the snapshot below.")
                    else:
                        st.caption("📹 This camera has no live video stream; snapshots only.")

                    if st.session_state["frame_source"] == "live" and "live_frame" in st.session_state:
                        frame = st.session_state["live_frame"]
                        live_stream_url = st.session_state.get("live_frame_stream_url")
                        live_cam_name = st.session_state.get("live_frame_cam_name")

                    # Auto-refresh uses st.fragment(run_every=...) — Streamlit's own
                    # mechanism for a piece of the app to re-run itself on a timer without
                    # blocking the rest of the session. An earlier version of this used a
                    # manual time.sleep()+st.rerun() poll loop instead; that blocked this
                    # session's script thread while waiting, which starved the browser's
                    # health-check connection and made the UI intermittently show "Is
                    # Streamlit still running?" — confirmed via live testing. st.fragment
                    # doesn't have that problem since it reruns independently.
                    if auto_refresh and frame is not None:
                        st.caption(f"🔴 Live — auto-refreshing this camera every {refresh_secs}s")

                        # A fragment's body also runs immediately whenever the full script
                        # reaches it, not just on its timer — and this one sits in the left
                        # column, which renders before the prediction column. Without the
                        # elapsed-time guard below, that first run would fetch + st.rerun()
                        # unconditionally, looping forever before the prediction section is
                        # ever reached (found via live testing: page stuck "running", frame
                        # never updating, nothing logged).
                        @st.fragment(run_every=f"{refresh_secs}s", key="live_camera_auto_refresh")
                        def _auto_refresh_tick(cam=cam, refresh_secs=refresh_secs):
                            last = st.session_state.setdefault("_last_auto_fetch_ts", time.time())
                            if time.time() - last < refresh_secs * 0.9:
                                return
                            st.session_state["_last_auto_fetch_ts"] = time.time()
                            try:
                                st.session_state["live_frame"] = load_frame(cam["image_url"])
                                st.session_state["live_frame_stream_url"] = cam.get("stream_url", "")
                                st.session_state["live_frame_cam_name"] = cam["name"]
                                st.session_state["live_frame_cam"] = cam
                                st.session_state["live_frame_cam_lat"] = cam.get("latitude")
                                st.session_state["live_frame_cam_lon"] = cam.get("longitude")
                            except Exception as e:
                                st.warning(f"Auto-refresh fetch failed, will retry: {e}")
                            st.rerun()  # default scope="app": refresh the whole page with the new frame

                        _auto_refresh_tick()
                else:
                    st.info("No cameras match that filter.")

        with tab_upload:
            uploaded = st.file_uploader("Traffic camera image", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                # file_uploader keeps returning the same file across reruns until
                # the user removes/replaces it; only treat it as a fresh action
                # (and switch the active source to it) the first time this
                # particular upload's id is seen.
                if st.session_state.get("_last_upload_id") != uploaded.file_id:
                    st.session_state["_last_upload_id"] = uploaded.file_id
                    st.session_state["frame_source"] = "upload"

                if st.session_state["frame_source"] == "upload":
                    image = Image.open(uploaded).convert("RGB")
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    live_stream_url = None  # an uploaded image has no associated live video stream
                    # ...and no camera identity either: tab_live runs first each rerun and can set
                    # live_cam_name from *last* rerun's frame_source before this block updates it —
                    # without this explicit reset a switch to Upload can still log under the
                    # previously-selected live camera's name (found via live testing).
                    live_cam_name = None

        with tab_url:
            url = st.text_input("Live camera snapshot URL (public DOT traffic cam, etc.)")
            if url:
                # Same idea as the uploader: the text box keeps returning the
                # same string across reruns, so only (re)fetch when it's new.
                if st.session_state.get("_last_seen_url") != url:
                    st.session_state["_last_seen_url"] = url
                    try:
                        st.session_state["url_frame"] = load_frame(url)
                        st.session_state["frame_source"] = "url"
                    except Exception as e:
                        st.error(f"Could not load frame: {e}")

                if st.session_state["frame_source"] == "url" and "url_frame" in st.session_state:
                    frame = st.session_state["url_frame"]
                    live_stream_url = None  # an arbitrary snapshot URL has no known video stream
                    live_cam_name = None  # explicit reset — see the comment in tab_upload above

        st.subheader("2. Conditions")
        st.caption(f"Predictions use California time, where the cameras are: {camera_now():%A %H:%M}.")
        c1, c2 = st.columns(2)
        with c1:
            temp_c = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, key="weather_temp")
            rain_1h = st.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0, key="weather_rain")
            snow_1h = st.number_input("Snow last hour (mm)", 0.0, 100.0, 0.0, key="weather_snow")
        with c2:
            clouds_all = st.slider("Cloud cover (%)", 0, 100, 20, key="weather_clouds")
            weather_main = st.selectbox("Weather", list(WEATHER_CODES.keys()), key="weather_main_select")
            is_holiday = st.checkbox("Holiday")

        model_choice = st.radio(
            "Model driving the forecast", ["lstm", "svr", "linear_regression"], horizontal=True,
            format_func=lambda m: f"{MODEL_LABELS[m]} (R²={MODEL_R2[m]})",
        )

        with st.expander("Advanced: live volume sensor reading (optional)"):
            st.caption("If this camera location also has an inductive-loop / radar volume "
                       "sensor, enter its recent readings for a more accurate prediction. "
                       "Leave blank to fall back to a seasonal (hour, day-of-week) average.")
            use_sensor = st.checkbox("I have real sensor readings")
            lag_1h = st.number_input("Volume 1h ago", value=0.0, disabled=not use_sensor) if use_sensor else None
            lag_3h = st.number_input("Volume 3h ago", value=0.0, disabled=not use_sensor) if use_sensor else None
            lag_24h = st.number_input("Volume 24h ago", value=0.0, disabled=not use_sensor) if use_sensor else None

    with col_result:
        st.subheader("3. Prediction")
        if not MODELS_READY:
            st.error(f"Trained model artifacts not found in {MODELS_DIR}. Run `python src/evaluate.py` first.")
        elif frame is None:
            st.info("Pick a live camera, upload an image, or provide a snapshot URL to run a prediction.")
        else:
            with st.spinner("Running vehicle detection..."):
                try:
                    cv_result = _cached_analyze_frame(frame)
                except Exception as e:
                    st.error(f"Vehicle detection failed: {e}")
                    st.stop()

            annotated_rgb = cv2.cvtColor(cv_result["annotated_frame"], cv2.COLOR_BGR2RGB)

            now = camera_now()  # California time: the cameras' clock, not this computer's
            try:
                predictions = _cached_predict_all(now.strftime("%Y-%m-%d %H:%M"), temp_c, rain_1h, snow_1h,
                                                   clouds_all, int(is_holiday), weather_main,
                                                   lag_1h, lag_3h, lag_24h)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.stop()

            cv_label = congestion_from_density(cv_result["density_ratio"])
            model_label = calibrated_label(predictions[model_choice])
            # The badge is what the camera shows right now. The historical model is trained on
            # Minnesota data, so on a quiet California road it often expects more traffic than
            # exists; letting it raise the badge made empty roads read "High".
            final_label = cv_label
            camera_name_for_log = live_cam_name or ("Uploaded image" if st.session_state.get("frame_source") == "upload"
                                                     else "Snapshot URL")

            # Only log once per genuinely new frame (not on every rerun caused
            # by e.g. dragging a weather slider) — otherwise the trend log
            # would fill up with near-duplicate entries from the same fetch.
            st.session_state.setdefault("_logged_for_frame_id", None)
            if st.session_state["_logged_for_frame_id"] != id(frame):
                log_prediction(camera_name_for_log, model_choice, predictions[model_choice],
                               model_label, cv_label, final_label)
                st.session_state["_logged_for_frame_id"] = id(frame)

            congestion_badge(final_label, key=camera_name_for_log)
            # Only a camera far busier than history expects suggests an incident; a camera quieter
            # than the (Minnesota) baseline is just a quiet road.
            if (detect_anomaly(model_label, cv_label)
                    and CONGESTION_LEVELS.index(cv_label) > CONGESTION_LEVELS.index(model_label)):
                st.warning("⚠️ **Unusual congestion detected** — the live camera shows much more traffic "
                          f"than history expects for this time ({cv_label} vs. an expected "
                          f"{model_label}). This kind of mismatch often means something out of the "
                          "ordinary is happening — an accident, event, or road closure — rather than "
                          "normal hour-to-hour variation.")
            st.image(annotated_rgb, caption="Detected vehicles", use_container_width=True)
            st.download_button(
                "⬇️ Download report (PNG)",
                data=build_report_png(cv_result["annotated_frame"], camera_name_for_log, final_label,
                                      cv_result["vehicle_count"], cv_result["density_ratio"],
                                      MODEL_LABELS[model_choice]),
                file_name=f"congestion_report_{now.strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )

            m1, m2 = st.columns(2)
            m1.markdown(counter_html(cv_result["vehicle_count"], "Vehicle count"), unsafe_allow_html=True)
            m2.metric("Image density ratio", cv_result["density_ratio"])
            st.caption(f"Live camera reading: **{cv_label}**  ·  "
                       f"{MODEL_LABELS[model_choice]} historical reading: **{model_label}**  ·  "
                       "the level shown is the live camera reading; the historical model is context "
                       "(it is trained on Minnesota traffic, so it can expect more than a quiet California road has).")

            st.markdown("##### All three models")
            max_volume = float(congestion_bins()[-1])
            comparison_df = pd.DataFrame([
                {"Model": MODEL_LABELS[name], "R²": MODEL_R2[name], "Predicted volume": round(predictions[name]),
                 "Level": calibrated_label(predictions[name])}
                for name in ("linear_regression", "svr", "lstm")
            ])
            st.dataframe(
                comparison_df, hide_index=True, use_container_width=True,
                column_config={
                    "Predicted volume": st.column_config.ProgressColumn(
                        "Predicted volume", min_value=0, max_value=max_volume, format="%d",
                    ),
                },
            )

            with st.expander("🔍 Why this prediction?"):
                X_row = build_feature_row(now, temp_c, rain_1h, snow_1h, clouds_all, int(is_holiday),
                                          weather_main, lag_1h=lag_1h, lag_3h=lag_3h, lag_24h=lag_24h)
                explanation = explain_prediction(model_choice, X_row)
                if explanation["kind"] == "exact":
                    st.caption(f"Exact breakdown for {MODEL_LABELS[model_choice]} — each feature's "
                              "contribution to the predicted volume, in the model's own scaled units.")
                    for item in explanation["top_features"]:
                        direction = "⬆️ increases" if item["contribution"] > 0 else "⬇️ decreases"
                        st.markdown(f"- **{item['label'].capitalize()}** {direction} the predicted "
                                    f"volume (contribution: {item['contribution']:+.2f})")
                else:
                    st.caption(f"{MODEL_LABELS[model_choice]} doesn't decompose into per-feature "
                              "contributions directly, so this instead shows which inputs are most "
                              "unusual right now compared to typical conditions for this dataset — "
                              "a proxy for what's driving an atypical prediction, not an exact breakdown.")
                    for item in explanation["top_features"]:
                        z = item["z_score"]
                        if abs(z) < 0.5:
                            st.markdown(f"- **{item['label'].capitalize()}** is close to typical")
                        else:
                            direction = "higher" if z > 0 else "lower"
                            st.markdown(f"- **{item['label'].capitalize()}** is unusually {direction} "
                                        f"than typical ({abs(z):.1f}σ)")

            camera_history = load_history(camera_name_for_log, limit=100)
            if len(camera_history) > 1:
                st.markdown("##### 📈 Recent history for this camera")
                hist_df = pd.DataFrame(camera_history)
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
                hist_chart = alt.Chart(hist_df).mark_line(point=True, color="#38bdf8").encode(
                    x=alt.X("timestamp:T", title="When checked"),
                    y=alt.Y("volume:Q", title="Predicted volume"),
                    tooltip=["timestamp:T", "volume:Q", "final_label:N"],
                ).properties(height=180).interactive()
                st.altair_chart(hist_chart, use_container_width=True)
                st.caption(f"{len(camera_history)} prediction(s) logged for this camera on this install "
                          "(logged once per fresh fetch, not on every slider tweak).")

            st.markdown("##### Today's forecast")
            forecast_df = _cached_forecast(now.strftime("%Y-%m-%d"), temp_c, rain_1h, snow_1h,
                                            clouds_all, int(is_holiday), weather_main, model_choice)
            st.altair_chart(forecast_chart(forecast_df, now.hour, congestion_bins(), mae=MODEL_MAE[model_choice]),
                             use_container_width=True)
            st.caption(f"Predicted volume across today using {MODEL_LABELS[model_choice]}, "
                       "weather held at the conditions set above. White dot marks the current hour; "
                       f"shaded band shows ±{MODEL_MAE[model_choice]:.0f} vehicles/hour, this model's "
                       "typical error on held-out test data.")

            with st.expander("🕒 When should I leave?"):
                st.caption("Uses today's forecast above to suggest a nearby hour with lower predicted "
                          "congestion, if one exists.")
                dl1, dl2 = st.columns(2)
                with dl1:
                    target_hour = st.slider("Planned departure hour", 0, 23, now.hour, key="depart_hour")
                with dl2:
                    flexibility = st.slider("How flexible are you? (± hours)", 1, 6, 2, key="depart_flex")
                rec = recommend_departure(forecast_df, target_hour, flexibility)
                if rec["improves"]:
                    st.success(f"Leaving at **{rec['best_hour']:02d}:00** instead of {target_hour:02d}:00 "
                              f"would drop you from **{rec['target_label']}** to **{rec['best_label']}** "
                              "congestion, based on today's forecast.")
                else:
                    st.info(f"**{target_hour:02d}:00** is already about as good as it gets within "
                            f"±{flexibility}h — predicted **{rec['target_label']}** congestion.")

            st.markdown("##### Speed check (experimental)")
            if not live_stream_url:
                st.caption("Only available for a live Caltrans camera (needs its video stream, not a static "
                           "image) — pick one from the Live Caltrans camera tab and fetch a snapshot first.")
            else:
                st.caption(
                    "Grabs a ~1.5s live video burst and tracks vehicles across frames to estimate speed. "
                    "There's no published camera calibration, so speed depends on the lane-width estimate "
                    "below — treat results as approximate, not a certified measurement."
                )
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    lane_width_px = st.number_input("Lane width in this frame (pixels)", min_value=5.0,
                                                    value=40.0, step=1.0,
                                                    help="Measure one traffic lane's width on the image above, in pixels.")
                with sc2:
                    lane_width_m = st.number_input("Real lane width (meters)", min_value=1.0,
                                                   value=US_LANE_WIDTH_M, step=0.1,
                                                   help="US standard freeway lane width is ~3.7m.")
                with sc3:
                    sc_info = (_limit_for_camera(st.session_state["live_frame_cam"])
                               if st.session_state.get("live_frame_cam") else None)
                    _apply_auto_limit(live_stream_url or "", sc_info, "speed_limit_single")
                    speed_limit_mph = st.number_input("Speed limit (mph)", min_value=5.0, value=65.0, step=5.0,
                                                      key="speed_limit_single")
                    sc_caption, sc_limit_source = _limit_caption(sc_info, speed_limit_mph)
                    st.caption(sc_caption)

                if st.button("Check speeds"):
                    meters_per_pixel = lane_width_m / lane_width_px
                    with st.spinner("Capturing live video burst and tracking vehicles..."):
                        try:
                            st.session_state["speed_result"] = check_speeds(live_stream_url, meters_per_pixel, speed_limit_mph)
                            st.session_state["speed_result_cam_name"] = live_cam_name
                            st.session_state["speed_result_limit_mph"] = speed_limit_mph
                            st.session_state["speed_result_limit_source"] = sc_limit_source
                        except Exception as e:
                            st.error(f"Speed check failed: {e}")
                            st.session_state.pop("speed_result", None)

                speed_result = st.session_state.get("speed_result")
                if speed_result is not None:
                    if not speed_result["results"]:
                        st.info("No vehicles could be tracked across the burst — try again "
                                "(traffic is dynamic) or pick a busier camera.")
                    else:
                        annotated_speed_frame = annotate_speeds(speed_result["frames"][-1], speed_result["results"])
                        st.image(cv2.cvtColor(annotated_speed_frame, cv2.COLOR_BGR2RGB),
                                  caption=f"{st.session_state.get('speed_result_cam_name')} — tracked vehicle speeds",
                                  use_container_width=True)
                        if st.button("🎨 Identify vehicle color & type"):
                            with st.spinner("Analyzing vehicle appearance (first run downloads a "
                                             "model, ~30s)..."):
                                st.session_state["vehicle_attrs"] = [
                                    describe_vehicle(speed_result["frames"][-1], r["last_box"])
                                    for r in speed_result["results"]
                                ]
                                st.session_state["vehicle_attrs_for"] = id(speed_result)

                        vehicle_attrs = None
                        if st.session_state.get("vehicle_attrs_for") == id(speed_result):
                            vehicle_attrs = st.session_state.get("vehicle_attrs")

                        rows = []
                        for i, r in enumerate(speed_result["results"]):
                            row = {"Vehicle": r["cls_name"], "Speed (mph)": round(r["speed_mph"], 1),
                                   "Speed (km/h)": round(r["speed_kmh"], 1),
                                   "Over limit (mph)": round(r["over_mph"], 1) if r["speeding"] else 0,
                                   "Speeding": "Yes" if r["speeding"] else "No"}
                            if vehicle_attrs is not None:
                                attrs = vehicle_attrs[i]
                                row["Color (best guess)"] = attrs["color"] or "unclear"
                                row["Body type (best guess)"] = (
                                    f"{attrs['type']} ({attrs['type_confidence']:.0%} conf.)"
                                    if attrs["type"] else "unclear"
                                )
                            rows.append(row)
                        speed_df = pd.DataFrame(
                            sorted(rows, key=lambda r: -r["Speed (mph)"])
                        )
                        st.dataframe(speed_df, hide_index=True, use_container_width=True)
                        if vehicle_attrs is not None:
                            st.caption("Color and body type are best-effort guesses from a general-purpose "
                                       "model (not trained on traffic cameras) — expect them to be wrong "
                                       "sometimes, especially at night or on small/distant vehicles.")
                        speeding_results = [r for r in speed_result["results"] if r["speeding"]]
                        if speeding_results:
                            st.warning(f"{len(speeding_results)} of {len(speed_result['results'])} tracked "
                                       f"vehicle(s) estimated over the "
                                       f"{st.session_state.get('speed_result_limit_mph'):.0f} mph limit.")

                            st.markdown("###### What the fine would have been (simulated)")
                            st.caption(ECHALLAN_DISCLAIMER)
                            veh_labels = [f"{r['cls_name'].title()} — {r['speed_mph']:.0f} mph "
                                          f"(+{r['over_mph']:.0f} over)" for r in speeding_results]
                            # Labels include each vehicle's speed, so a fresh Check speeds run
                            # changes them and can leave a stale label string behind (see
                            # camera_picker_map).
                            if "echallan_vehicle" in st.session_state and not _is_valid_index(
                                    st.session_state["echallan_vehicle"], len(speeding_results)):
                                st.session_state["echallan_vehicle"] = 0
                            veh_choice = st.selectbox("Vehicle", range(len(speeding_results)),
                                                      format_func=lambda i: veh_labels[i], key="echallan_vehicle")
                            if not isinstance(veh_choice, int):
                                st.rerun()

                            ec1, ec2 = st.columns([1, 2])
                            with ec1:
                                if st.button("🔍 Try plate recognition"):
                                    with st.spinner("Reading plate (first run downloads an OCR model, ~30s)..."):
                                        plate_result = read_plate(speed_result["frames"][-1],
                                                                  speeding_results[veh_choice]["last_box"])
                                    st.session_state["_echallan_ocr_text"] = plate_result["text"]
                                    st.session_state["_echallan_ocr_conf"] = plate_result["confidence"]
                                    if plate_result["text"]:
                                        st.session_state["echallan_plate"] = plate_result["text"]
                            with ec2:
                                vehicle_number = st.text_input("Vehicle number (optional)", key="echallan_plate")

                            if st.session_state.get("_echallan_ocr_text"):
                                st.caption(f"✅ Auto-detected: {st.session_state['_echallan_ocr_text']} "
                                          f"(confidence {st.session_state.get('_echallan_ocr_conf', 0):.2f}) — "
                                          "unverified, double-check before using; edit above if wrong.")
                            elif "_echallan_ocr_conf" in st.session_state:
                                st.caption("⚠️ Could not read a plate automatically from this frame — a common "
                                          "outcome on this footage (nighttime blur, occlusion, viewing angle; "
                                          "see README). Enter the vehicle number manually if known.")

                            is_auto = bool(st.session_state.get("_echallan_ocr_text")) and \
                                vehicle_number == st.session_state.get("_echallan_ocr_text")
                            plate_conf = st.session_state.get("_echallan_ocr_conf", 0.0) if is_auto else 0.0

                            if st.button("Generate summary"):
                                challan = generate_challan(
                                    speeding_results[veh_choice],
                                    st.session_state.get("speed_result_cam_name") or "Unknown camera",
                                    st.session_state.get("speed_result_limit_mph"),
                                    vehicle_number=vehicle_number,
                                    plate_auto_detected=is_auto,
                                    plate_confidence=plate_conf,
                                    limit_source=st.session_state.get("speed_result_limit_source", ""),
                                )
                                append_to_log(challan)
                                st.session_state["last_challan_html"] = render_challan_html(challan)

                            if st.session_state.get("last_challan_html"):
                                st.markdown(st.session_state["last_challan_html"], unsafe_allow_html=True)

                            with st.expander(f"Challan history ({len(load_log())} issued this install)"):
                                log = load_log()
                                if log:
                                    st.dataframe(
                                        pd.DataFrame(log)[["challan_id", "issued_at", "camera_name", "vehicle_class",
                                                          "vehicle_number", "plate_auto_detected", "speed_mph", "fine_inr"]],
                                        hide_index=True, use_container_width=True,
                                    )
                                else:
                                    st.caption("No challans generated yet.")
                        else:
                            st.caption("No tracked vehicle is over the limit, so there is no challan to simulate. "
                                       "You can also use the 🧾 Live e-Challan tab, which watches a camera and issues challans automatically.")

# ---------------------------------------------------------------- monitoring grid
with mode_grid:
    st.subheader("Monitor several cameras at once")
    st.caption("Pick a district and how many cameras to check — each gets its own live "
               "vehicle detection and congestion badge in one view.")

    if not MODELS_READY:
        st.error(f"Trained model artifacts not found in {MODELS_DIR}. Run `python src/evaluate.py` first.")
    else:
        g1, g2 = st.columns([1, 1])
        with g1:
            grid_district = st.selectbox("District", DISTRICTS, index=6, key="grid_district",
                                         format_func=lambda d: f"District {d}")
        with g2:
            grid_search = st.text_input("Filter by route / location", key="grid_search")

        try:
            grid_cameras = _cached_cameras(grid_district)
        except Exception as e:
            grid_cameras = []
            st.error(f"Could not reach Caltrans camera feed: {e}")

        if grid_search:
            needle = grid_search.lower()
            grid_cameras = [c for c in grid_cameras if needle in c["name"].lower() or needle in c["nearby_place"].lower()]

        max_cams = min(len(grid_cameras), 16)
        if max_cams == 0:
            st.info("No cameras match that filter.")
        else:
            n_cams = st.slider("Number of cameras to check", 1, max_cams, min(6, max_cams))

            with st.expander("Weather assumptions (applied to every camera in the grid)"):
                gw1, gw2 = st.columns(2)
                with gw1:
                    g_temp = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, key="grid_temp")
                    g_rain = st.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0, key="grid_rain")
                with gw2:
                    g_clouds = st.slider("Cloud cover (%)", 0, 100, 20, key="grid_clouds")
                    g_weather = st.selectbox("Weather", list(WEATHER_CODES.keys()), key="grid_weather")

            if st.button("Load grid", type="primary"):
                selected_cams = grid_cameras[:n_cams]
                now = camera_now()
                try:
                    model_pred = predict_all_models(now, g_temp, g_rain, 0.0, g_clouds, 0, g_weather)
                    hist_label = calibrated_label(model_pred["lstm"])
                except Exception as e:
                    st.error(f"Historical model prediction failed: {e}")
                    hist_label = None

                results = []
                progress = st.progress(0.0, text="Fetching and analyzing cameras...")
                for i, c in enumerate(selected_cams):
                    try:
                        cam_frame = load_frame(c["image_url"])
                        cam_result = analyze_frame(cam_frame)
                        cam_cv_label = congestion_from_density(cam_result["density_ratio"])
                        # Badge = this camera's own live reading. The historical model's label is the
                        # same for every camera, so fusing it in here would floor them all at it
                        # (an empty road showed "Moderate"). It is shown once, as a caption, below.
                        results.append({"camera": c, "result": cam_result, "cv_label": cam_cv_label,
                                        "final_label": cam_cv_label, "error": None})
                    except Exception as e:
                        results.append({"camera": c, "result": None, "cv_label": None,
                                        "final_label": None, "error": str(e)})
                    progress.progress((i + 1) / len(selected_cams), text=f"Analyzed {i + 1}/{len(selected_cams)}")
                progress.empty()
                st.session_state["grid_results"] = results
                st.session_state["grid_hist_label"] = hist_label

            grid_results = st.session_state.get("grid_results")
            if grid_results:
                grid_hist = st.session_state.get("grid_hist_label")
                if grid_hist:
                    st.caption(f"Each badge is that camera's own live reading (vehicle density). The historical "
                               f"model for this time and weather says **{grid_hist}** for every camera; the "
                               "Single camera tab shows it next to the live reading.")
                severity_counts = pd.Series(
                    [r["final_label"] for r in grid_results if r["final_label"]]
                ).value_counts().reindex(CONGESTION_LEVELS, fill_value=0)
                summary_cols = st.columns(4)
                for level, col in zip(CONGESTION_LEVELS, summary_cols):
                    col.metric(level, int(severity_counts[level]))

                st.divider()
                cards_per_row = 3
                for row_start in range(0, len(grid_results), cards_per_row):
                    row = grid_results[row_start: row_start + cards_per_row]
                    cols = st.columns(cards_per_row)
                    for col, item in zip(cols, row):
                        c = item["camera"]
                        with col:
                            st.markdown(f"**{c['name']}**")
                            st.caption(c["nearby_place"])
                            if item["error"]:
                                st.warning(f"Could not fetch: {item['error']}")
                            else:
                                annotated_rgb = cv2.cvtColor(item["result"]["annotated_frame"], cv2.COLOR_BGR2RGB)
                                st.image(annotated_rgb, use_container_width=True)
                                congestion_badge(item["final_label"], small=True)
                                st.caption(f"{item['result']['vehicle_count']} vehicles · "
                                           f"density {item['result']['density_ratio']}")

# ---------------------------------------------------------------- route congestion
ROUTE_MODES = ["Pick cameras", "Search places", "Click the map"]


def _run_route_check(cams, weather, route=None, endpoints=None, in_route_order=False):
    """Check every camera in `cams` (live frame, vehicle detection, fused with the
    historical model) and store the outcome for _show_route_view()."""
    temp, rain, clouds, weather_main = weather
    try:
        model_pred = predict_all_models(camera_now(), temp, rain, 0.0, clouds, 0, weather_main)
        hist_label = calibrated_label(model_pred["lstm"])
    except Exception as e:
        st.error(f"Historical model prediction failed: {e}")
        hist_label = None

    results = []
    if cams:
        progress = st.progress(0.0, text="Checking route segments...")
        for i, c in enumerate(cams):
            try:
                cam_result = analyze_frame(load_frame(c["image_url"]))
                cam_cv_label = congestion_from_density(cam_result["density_ratio"])
                # Per-segment label is the camera's own live reading. The historical model gives one
                # time/weather prediction shared by every camera, so fusing it in per segment would put
                # the same floor under all of them (e.g. every segment "Moderate" regardless of vehicles).
                # It is applied once, to the overall reading, in _show_route_view().
                results.append({"camera": c, "final_label": cam_cv_label,
                                "vehicle_count": cam_result["vehicle_count"], "error": None})
            except Exception as e:
                results.append({"camera": c, "final_label": None, "vehicle_count": None, "error": str(e)})
            progress.progress((i + 1) / len(cams), text=f"Checked {i + 1}/{len(cams)}")
        progress.empty()
    st.session_state["route_view"] = {"results": results, "route": route, "endpoints": endpoints,
                                      "in_route_order": in_route_order, "hist_label": hist_label}


def _check_between(start, end, weather):
    """Route between two (lat, lon) points: fetch the road path, find the cameras
    along it, and check them."""
    try:
        with st.spinner("Finding the road route..."):
            route = _cached_route(tuple(start), tuple(end))
    except Exception as e:
        st.error(f"Couldn't get a driving route between those two points: {e}")
        return
    with st.spinner("Looking for traffic cameras along the route (the first search loads every "
                    "Caltrans district, which takes a moment)..."):
        cams = cameras_near_route(route["points"], _all_cameras())
    _run_route_check(cams, weather, route=route, endpoints=(tuple(start), tuple(end)), in_route_order=True)


def _show_route_view():
    view = st.session_state.get("route_view")
    if not view:
        return
    results, route, endpoints = view["results"], view["route"], view["endpoints"]

    valid = [r for r in results if r["final_label"]]
    if valid:
        live_worst = max((r["final_label"] for r in valid), key=CONGESTION_LEVELS.index)
        worst_segment = next(r for r in valid if r["final_label"] == live_worst)
        hist_label = view.get("hist_label")
        route_label = live_worst  # the cameras' live readings; history is shown as context only
        st.markdown("###### Overall route congestion")
        congestion_badge(route_label)
        if hist_label:
            st.caption(f"Worst live camera: **{live_worst}** ({worst_segment['camera']['name']}). "
                       f"Historical model for this time and weather: **{hist_label}**. "
                       "The overall reading is the worst live camera; history is a Minnesota-trained "
                       "baseline, shown for context only.")
        else:
            st.caption(f"Driven by the worst segment: **{worst_segment['camera']['name']}**")
    elif results:
        st.warning("Couldn't read any segment on this route right now.")
    else:
        st.info(f"No Caltrans cameras within {CAMERA_ON_ROUTE_M / 1000:.1f} km of this route, so there is "
                "no congestion reading to show. Caltrans cameras only cover California highways.")

    if view["in_route_order"]:
        ordered = results  # already in the order they're met along the road
    else:
        # The multiselect hands cameras back in list order, not drive order, so put
        # them in corridor order; cameras with no coordinates can't be placed and go last.
        located = [r for r in results if r["camera"].get("latitude") and r["camera"].get("longitude")]
        by_camera = {id(r["camera"]): r for r in located}
        ordered = [by_camera[id(c)] for c in order_cameras([r["camera"] for r in located])] if located else []
        ordered += [r for r in results if id(r["camera"]) not in by_camera]

    st.markdown("###### Route map")
    _route_map(ordered, route, endpoints)

    if ordered:
        st.markdown("###### Segment-by-segment")
    for n, r in enumerate(ordered, start=1):
        seg_cols = st.columns([3, 1, 1])
        seg_cols[0].markdown(f"**{n}. {r['camera']['name']}** — {r['camera']['nearby_place']}")
        if r["error"]:
            seg_cols[1].caption(f"⚠️ {r['error']}")
        else:
            seg_cols[1].markdown(f"{r['vehicle_count']} vehicles")
            color = LEVEL_COLORS[r["final_label"]]
            seg_cols[2].markdown(
                f"<span style='color:{color};font-weight:700'>{r['final_label']}</span>",
                unsafe_allow_html=True,
            )


with mode_route:
    st.subheader("Route congestion")

    if not MODELS_READY:
        st.error(f"Trained model artifacts not found in {MODELS_DIR}. Run `python src/evaluate.py` first.")
    else:
        route_mode = st.radio("Build the route by", ROUTE_MODES, horizontal=True, key="route_mode")

        with st.expander("Weather assumptions (applied to every camera on the route)"):
            rw1, rw2 = st.columns(2)
            with rw1:
                rt_temp = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, key="route_temp")
                rt_rain = st.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0, key="route_rain")
            with rw2:
                rt_clouds = st.slider("Cloud cover (%)", 0, 100, 20, key="route_clouds")
                rt_weather = st.selectbox("Weather", list(WEATHER_CODES.keys()), key="route_weather")
        weather = (rt_temp, rt_rain, rt_clouds, rt_weather)

        if route_mode == ROUTE_MODES[0]:
            st.caption("Pick a district, then select every camera along your route (any order) to get one "
                       "overall reading, always driven by the worst segment, so a single jammed stretch is "
                       "never hidden by an otherwise-clear route.")
            rt1, rt2 = st.columns([1, 1])
            with rt1:
                route_district = st.selectbox("District", DISTRICTS, index=6, key="route_district",
                                              format_func=lambda d: f"District {d}")
            with rt2:
                route_search = st.text_input("Filter by route / location", key="route_search")

            try:
                route_cameras = _cached_cameras(route_district)
            except Exception as e:
                route_cameras = []
                st.error(f"Could not reach Caltrans camera feed: {e}")

            if route_search:
                needle = route_search.lower()
                route_cameras = [c for c in route_cameras
                                 if needle in c["name"].lower() or needle in c["nearby_place"].lower()]

            if not route_cameras:
                st.info("No cameras match that filter.")
            else:
                route_labels = [f"{c['name']} — {c['nearby_place']}" for c in route_cameras]
                # Same stale-label issue as camera_picker_map: after a district or filter
                # change, previously chosen cameras that aren't in the new list come back as
                # label strings, which would crash `route_cameras[i]` below. Drop them.
                previous = st.session_state.get("route_choice", [])
                kept = [i for i in previous if _is_valid_index(i, len(route_cameras))]
                if kept != previous:
                    st.session_state["route_choice"] = kept
                chosen_idx = st.multiselect("Cameras along your route (pick 2 or more)", range(len(route_cameras)),
                                            format_func=lambda i: route_labels[i], key="route_choice")
                if not all(isinstance(i, int) for i in chosen_idx):
                    st.rerun()

                if len(chosen_idx) < 2:
                    st.info("Select at least 2 cameras to build a route.")
                elif st.button("Check route", type="primary", key="route_check_cameras"):
                    _run_route_check([route_cameras[i] for i in chosen_idx], weather)

        elif route_mode == ROUTE_MODES[1]:
            st.caption("Type a start and a destination (any place OpenStreetMap knows). The app finds the "
                       "driving route and checks every Caltrans camera along it. Cameras only cover "
                       "California highways, so stretches elsewhere stay gray.")
            pc1, pc2 = st.columns(2)
            start_q = pc1.text_input("Start", key="route_start_q", placeholder="e.g. Burbank Airport")
            end_q = pc2.text_input("Destination", key="route_end_q", placeholder="e.g. Santa Monica Pier")
            if st.button("Find places", key="route_find_places"):
                try:
                    with st.spinner("Searching OpenStreetMap..."):
                        st.session_state["geo_start"] = geocode(start_q)
                        time.sleep(1.1)  # Nominatim's usage policy: at most one request per second
                        st.session_state["geo_end"] = geocode(end_q)
                except Exception as e:
                    st.error(f"Place search failed: {e}")

            geo_start, geo_end = st.session_state.get("geo_start"), st.session_state.get("geo_end")
            if geo_start is None or geo_end is None:
                st.info("Enter a start and a destination, then click Find places.")
            elif not geo_start or not geo_end:
                missing = " and ".join(n for n, g in (("the start", geo_start), ("the destination", geo_end)) if not g)
                st.warning(f"No match found for {missing}. Try a more specific name, such as adding the city.")
            else:
                s_names, e_names = [p["name"] for p in geo_start], [p["name"] for p in geo_end]
                # Options are the labels themselves; reset a leftover choice from an earlier search
                # that isn't in the new list (see camera_picker_map for why stale labels bite).
                for key, names in (("route_start_pick", s_names), ("route_end_pick", e_names)):
                    if st.session_state.get(key) not in names:
                        st.session_state[key] = names[0]
                s_pick = st.selectbox("Start: pick the right match", s_names, key="route_start_pick")
                e_pick = st.selectbox("Destination: pick the right match", e_names, key="route_end_pick")
                s_place, e_place = geo_start[s_names.index(s_pick)], geo_end[e_names.index(e_pick)]
                if st.button("Check route", type="primary", key="route_check_places"):
                    _check_between((s_place["lat"], s_place["lon"]), (e_place["lat"], e_place["lon"]), weather)

        else:
            st.caption("Click the map to drop the start, then click again for the destination (a third click "
                       "starts over with a new start). The app finds the driving route and checks every "
                       "Caltrans camera along it.")
            pts = st.session_state.setdefault("route_click_pts", {"start": None, "end": None, "next": "start"})
            view_center, view_zoom = st.session_state.get("route_click_view", ((34.05, -118.25), 9))

            fmap = folium.Map(location=view_center, zoom_start=view_zoom, tiles="OpenStreetMap")  # Carto's raster tiles now demand an API key
            if pts["start"]:
                folium.Marker(pts["start"], tooltip="Start",
                              icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fmap)
            if pts["end"]:
                folium.Marker(pts["end"], tooltip="Destination",
                              icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(fmap)
            map_out = st_folium(fmap, height=420, use_container_width=True, key="route_click_map",
                                returned_objects=["last_clicked", "center", "zoom"],
                                center=view_center, zoom=view_zoom)

            if map_out:
                if map_out.get("center") and map_out.get("zoom"):
                    st.session_state["route_click_view"] = (
                        (map_out["center"]["lat"], map_out["center"]["lng"]), map_out["zoom"])
                click = map_out.get("last_clicked")
                if click:
                    click_key = (round(click["lat"], 6), round(click["lng"], 6))
                    # The component keeps returning its last click on every rerun, so only act on a new one.
                    if st.session_state.get("_route_last_click") != click_key:
                        st.session_state["_route_last_click"] = click_key
                        if pts["next"] == "start":
                            pts["start"], pts["end"], pts["next"] = (click["lat"], click["lng"]), None, "end"
                        else:
                            pts["end"], pts["next"] = (click["lat"], click["lng"]), "start"
                        st.rerun()

            status = ("Click the map to set the **start**." if pts["next"] == "start" and not pts["start"]
                      else "Click the map to set the **destination**." if pts["next"] == "end"
                      else "Both points set. Check the route, or click again to start over.")
            st.caption(f"{status}  Start: {'set' if pts['start'] else 'not set'} · "
                       f"Destination: {'set' if pts['end'] else 'not set'}")
            cc1, cc2 = st.columns([1, 1])
            if cc2.button("Clear points", key="route_clear_points"):
                # Deliberately leaves _route_last_click alone: the map component still returns its
                # old click, which would otherwise re-add the point just cleared.
                st.session_state["route_click_pts"] = {"start": None, "end": None, "next": "start"}
                st.rerun()
            if pts["start"] and pts["end"] and cc1.button("Check route", type="primary", key="route_check_click"):
                _check_between(pts["start"], pts["end"], weather)

        _show_route_view()


# ---------------------------------------------------------------- live monitor
def _render_live_monitor():
    """Polled once a second while a monitor runs: shows the annotated live frame, status and challans."""
    mon = st.session_state.get("live_monitor")
    if mon is None:
        st.info("Pick a camera and press **Start monitoring**. The app will watch the live video, detect and "
                "track the vehicles, estimate their speeds, and generate a simulated challan for any vehicle "
                "clearly over the limit.")
        return
    snap = mon.snapshot()
    status = snap["status"]
    left, right = st.columns([1.7, 1])
    with left:
        if snap["frame"] is not None:
            st.image(cv2.cvtColor(snap["frame"], cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption=f"{mon.camera_name}: live detection (boxes show estimated speed in mph once "
                             "calibrated; gray = no speed estimate)")
        elif status in ("starting", "calibrating"):
            st.info("Opening the stream and loading the detector... (the first frame can take 10-20 seconds)")
    with right:
        st.caption(f"Status: **{status}**  ·  {snap['elapsed']:.0f}s of {mon.run_seconds:.0f}s  ·  "
                   f"{snap['fps']:.1f} frames/s analyzed")
        if snap["message"]:
            (st.error if status == "error" else st.warning)(snap["message"])
        if snap["congestion"]:
            congestion_badge(snap["congestion"], key=mon.camera_name)
            st.markdown(counter_html(snap["vehicle_count"], "Vehicles in view right now"), unsafe_allow_html=True)
        ss = snap["speed_status"]
        if ss == "ok":
            st.success(f"Speed tracking on ({snap['tracks_with_speed']} vehicles tracked, calibration scale "
                       f"x{snap['scale']:.2f}).")
            if snap["scale"] > 1.8 or snap["scale"] < 0.55:
                st.caption("⚠️ A large correction was needed to match the typical traffic speed you set, so "
                           "these speeds are mostly *relative to the flow of traffic* (who is faster than "
                           "the rest), not absolute mph.")
        elif ss == "calibrating":
            st.info("Calibrating speed tracking: needs about 25 seconds of video and a dozen tracked "
                    f"vehicles ({snap['tracks_with_speed']} so far).")
        elif ss == "unreliable":
            st.warning("Speeds are not reliable on this camera right now, so no challans are issued. Detection "
                       "and counting still work. Try a camera with a clear view along a straight road.")
        if snap["calibration"]:
            st.caption(f"Camera model fitted from the traffic: estimated camera height "
                       f"{snap['calibration']['camera_height_m']} m, fit quality "
                       f"{snap['calibration']['correlation']}.")

    events = snap["events"]
    st.markdown(f"###### Simulated challans from this session ({len(events)})")
    if not events:
        st.caption("None yet. A vehicle must stay over the limit plus the margin for several consecutive "
                   "measurements before a challan is generated.")
    for ev in reversed(events):
        ec1, ec2 = st.columns([1, 2])
        ec1.image(cv2.cvtColor(ev["crop"], cv2.COLOR_BGR2RGB), caption=f"{ev['cls_name']} at {ev['time']}, "
                  f"~{ev['speed_mph']:.0f} mph", use_container_width=True)
        if ev["challan"] is not None:
            ec2.markdown(render_challan_html(ev["challan"]), unsafe_allow_html=True)
        else:
            ec2.info("Generating challan (reading plate / identifying the vehicle)..." if ev["status"] == "processing"
                     else f"Could not generate the challan: {ev['status']}")

    if not mon.running and st.session_state.get("_live_final_rerun") != id(mon):
        st.session_state["_live_final_rerun"] = id(mon)
        st.rerun()  # one full rerun so polling stops now that the monitor has finished


with mode_live:
    st.subheader("Live e-Challan: watch a camera, detect cars, issue simulated challans")
    st.warning(ECHALLAN_DISCLAIMER)
    with st.expander("How this works, and how far to trust it"):
        st.markdown(
            "- The app reads the camera's **live video stream**, runs YOLO on about 4 frames per second, and "
            "tracks each vehicle from frame to frame.\n"
            "- **Speed** comes from a flat-road camera model fitted from the cars themselves (no manual "
            "calibration). The camera's zoom is unknown, so speeds are also scaled so the median vehicle matches "
            "the *typical traffic speed* you set below. Speeds are only estimated for vehicles in a reliable "
            "distance band, and if the camera model is not convincing, speeds and challans are switched off.\n"
            "- **Treat speeds as rough estimates** (easily 20-30% off, and not validated against radar). "
            "A challan here is a demonstration of the pipeline, never evidence of a real violation.\n"
            "- A challan is generated automatically from the real vehicle observed: its type, estimated speed, "
            "time and camera, plus its plate (if readable) and color/type (if enabled). Pictures stay in memory "
            "only; the on-disk challan log holds text fields only."
        )

    lm1, lm2 = st.columns([1, 1])
    with lm1:
        lm_district = st.selectbox("District", DISTRICTS, index=6, key="live_district",
                                   format_func=lambda d: f"District {d}")
    with lm2:
        lm_filter = st.text_input("Filter by route / location", key="live_filter",
                                  placeholder="e.g. I-5, 405, Burbank")
    try:
        lm_cams = [c for c in _cached_cameras(lm_district) if is_playable_url(c.get("stream_url", ""))]
    except Exception as e:
        lm_cams = []
        st.error(f"Could not reach Caltrans camera feed: {e}")
    if lm_filter:
        needle = lm_filter.lower()
        lm_cams = [c for c in lm_cams if needle in c["name"].lower() or needle in c["nearby_place"].lower()]

    if not lm_cams:
        st.info("No cameras with a live video stream match that filter.")
    else:
        lm_labels = [f"{c['name']} — {c['nearby_place']}" for c in lm_cams]
        if st.session_state.get("live_camera") not in lm_labels:
            # Default to a camera known to calibrate well (a clear straight view with cars at many
            # distances) when it is in this list, so the first run shows the whole pipeline working.
            preferred = [l for l in lm_labels if "(37) Olive" in l]
            st.session_state["live_camera"] = (preferred or lm_labels)[0]
        lm_pick = st.selectbox(f"Camera ({len(lm_cams)} with video)", lm_labels, key="live_camera")
        st.caption("Tip: speed tracking works best on a camera that looks straight down a road with cars at "
                   "many distances (I-5 at Olive, District 7, is a good one). On a camera with a sideways or "
                   "unusual view it says speeds are not reliable and issues no challans, on purpose.")
        lm_cam = lm_cams[lm_labels.index(lm_pick)]

        if st.checkbox("📹 Show this camera's live video", value=True, key="live_show_video",
                       help="Plays straight from Caltrans in your browser. Some streams are offline."):
            components.html(hls_player_html(lm_cam["stream_url"], height=340), height=352)
            st.caption("This is the camera's real-time video. After you press Start monitoring, the annotated "
                       "feed below (boxes, speeds, challans) updates about once a second.")

        lm_auto = st.checkbox("Set the speed limit automatically from the camera's road (OpenStreetMap)",
                              value=True, key="live_auto_limit")
        lm_info = _limit_for_camera(lm_cam) if lm_auto else None
        if lm_auto:
            st.session_state["_limit_cam_for_poll"] = lm_cam
            if st.session_state.get("_limit_pending"):
                _limit_poller()
        if lm_auto:
            _apply_auto_limit(lm_cam["stream_url"], lm_info, "live_limit", "live_ref")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lm_limit = lc1.number_input("Speed limit (mph)", min_value=5.0, value=65.0, step=5.0, key="live_limit")
        lm_margin = lc2.number_input("Challan only if over by (mph)", min_value=1.0, value=15.0, step=1.0,
                                     key="live_margin", help="Absorbs the speed estimate's error.")
        lm_ref = lc3.number_input("Typical traffic speed here (mph)", min_value=10.0, value=60.0, step=5.0,
                                  key="live_ref", help="Speeds are scaled so the median tracked vehicle "
                                  "matches this. Set it to roughly how fast traffic normally moves here.")
        lm_secs = lc4.slider("Run for (seconds)", 30, 300, 120, step=10, key="live_secs")
        lo1, lo2 = st.columns(2)
        lm_plates = lo1.checkbox("Try to read license plates (slow; plates are rarely legible)", key="live_plates")
        lm_ident = lo2.checkbox("Identify color and type (slow; first run downloads a model)", key="live_ident")

        lm_caption, lm_limit_source = _limit_caption(lm_info, lm_limit) if lm_auto else ("", "entered manually")
        if lm_caption:
            st.caption(lm_caption)
        if lm_limit + lm_margin < lm_ref:
            st.caption("⚠️ The limit plus margin is below the typical traffic speed you set, so most vehicles "
                       "will count as violations. Raise the limit or the margin for a realistic run "
                       "(a session issues at most 12 challans, at least 3 seconds apart).")
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("▶ Start monitoring", type="primary", key="live_start"):
            old = st.session_state.get("live_monitor")
            if old is not None:
                old.stop()
            new_mon = LiveMonitor(lm_cam["stream_url"], f"{lm_cam['name']} ({lm_cam['nearby_place']})",
                                  limit_mph=lm_limit, margin_mph=lm_margin, reference_mph=lm_ref,
                                  run_seconds=float(lm_secs), read_plates=lm_plates, identify=lm_ident,
                                  limit_source=lm_limit_source)
            new_mon.start()
            st.session_state["live_monitor"] = new_mon
            st.session_state.pop("_live_final_rerun", None)
        if b2.button("■ Stop", key="live_stop"):
            mon_now = st.session_state.get("live_monitor")
            if mon_now is not None:
                mon_now.stop()

    _mon_now = st.session_state.get("live_monitor")
    st.fragment(run_every=1.0 if (_mon_now is not None and _mon_now.running) else None)(_render_live_monitor)()


# ---------------------------------------------------------------- e-challan
with mode_challan:
    st.divider()
    st.subheader("Other ways to generate a challan, and the history")
    st.caption("A challan is the document an over-speeding vehicle would receive: live video, vehicle "
               "tracking, speed estimate, optional plate reading, then a challan showing what the fine "
               "would have been. Besides the live monitor above:")
    st.markdown("- **From a speed-check burst:** 🔍 Single camera, live camera, **Speed check**. If a "
                "vehicle is over the limit, its challan form appears there.\n"
                "- **From made-up values (below):** a way to demo the challan itself without waiting for a "
                "real speeder.")

    with st.form("manual_challan"):
        mc1, mc2 = st.columns(2)
        manual_camera = mc1.text_input("Camera / location", value="Demo camera")
        manual_plate = mc2.text_input("Vehicle number (optional)")
        mc3, mc4, mc5 = st.columns(3)
        manual_class = mc3.selectbox("Vehicle type", list(FINE_SCHEDULE_INR), index=list(FINE_SCHEDULE_INR).index("car"))
        manual_limit = mc4.number_input("Speed limit (mph)", min_value=5.0, value=65.0, step=5.0)
        manual_speed = mc5.number_input("Recorded speed (mph)", min_value=0.0, value=85.0, step=1.0)
        generate_manual = st.form_submit_button("Generate demo challan", type="primary")

    if generate_manual:
        if manual_speed <= manual_limit:
            st.info("That speed is not over the limit, so no challan is generated. "
                    "A challan only exists for a vehicle that was over the limit.")
        else:
            manual_result = {"speeding": True, "cls_name": manual_class, "speed_mph": manual_speed,
                             "over_mph": manual_speed - manual_limit}
            manual_challan = generate_challan(manual_result, f"{manual_camera} (manual demo)", manual_limit,
                                              vehicle_number=manual_plate, source="manual")
            append_to_log(manual_challan)
            st.session_state["manual_challan_html"] = render_challan_html(manual_challan)

    if st.session_state.get("manual_challan_html"):
        st.markdown(st.session_state["manual_challan_html"], unsafe_allow_html=True)

    history = load_log()
    st.markdown(f"###### Challan history ({len(history)} generated on this install)")
    if history:
        history_df = pd.DataFrame(history).reindex(columns=[
            "challan_id", "issued_at", "source", "camera_name", "vehicle_class", "vehicle_description",
            "vehicle_number", "plate_auto_detected", "speed_mph", "fine_inr"])
        st.dataframe(history_df, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download history (CSV)", history_df.to_csv(index=False).encode("utf-8"),
                           file_name="challan_history.csv", mime="text/csv")
        with st.expander("Clear history"):
            st.caption("Deletes the local challan log (demo data only) from this computer.")
            confirmed = st.checkbox("Yes, delete all stored challans", key="confirm_clear_challans")
            if confirmed and st.button("Delete challan history", key="clear_challans"):
                st.success(f"Deleted {clear_log()} challans.")
                st.session_state.pop("manual_challan_html", None)
                st.rerun()
    else:
        st.caption("No challans generated yet.")
