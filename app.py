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
from datetime import datetime

import altair as alt
import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cv_module import analyze_frame, congestion_from_density, load_frame  # noqa: E402
from data_loader import WEATHER_CODES  # noqa: E402
from live_cameras import DISTRICTS, fetch_cameras  # noqa: E402
from live_predict import (  # noqa: E402
    CONGESTION_LEVELS, MODELS_DIR, build_feature_row, calibrated_label, congestion_bins,
    detect_anomaly, forecast_day, fuse_labels, predict_all_models, recommend_departure,
)
from speed_estimation import US_LANE_WIDTH_M, annotate_speeds, check_speeds  # noqa: E402
from echallan import DISCLAIMER as ECHALLAN_DISCLAIMER, append_to_log, generate_challan, load_log, render_challan_html  # noqa: E402
from anpr import read_plate  # noqa: E402
from vehicle_attributes import describe_vehicle  # noqa: E402
from prediction_log import load_history, log_prediction  # noqa: E402
from explain import explain_prediction  # noqa: E402
from weather import fetch_current_weather  # noqa: E402
from report import build_report_png  # noqa: E402

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


def congestion_badge(level: str, small: bool = False):
    color = LEVEL_COLORS[level]
    pad, title_size, level_size = ("10px 14px", "11px", "22px") if small else ("18px 24px", "13px", "44px")
    st.markdown(
        f"""<div style="background:{color}1a;border:2px solid {color};border-radius:14px;
        padding:{pad};text-align:center;margin-bottom:10px;">
        <div style="font-size:{title_size};color:{color};font-weight:700;letter-spacing:2px;
        text-transform:uppercase;">Congestion</div>
        <div style="font-size:{level_size};color:{color};font-weight:800;line-height:1.2;">{level}</div>
        </div>""",
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
    if st.session_state[selected_idx_key] >= len(filtered):
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
    highlight_layer = pdk.Layer(
        "ScatterplotLayer", id="selected-camera", data=selected_row,
        get_position=["lon", "lat"],
        get_color="[239, 68, 68, 230]",
        get_radius=400,
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


st.set_page_config(page_title="Smart Traffic Congestion Prediction", layout="wide", page_icon="🚦")
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Traffic Congestion Prediction")
st.caption("Computer vision vehicle counting + Linear Regression / SVR / LSTM, fused with real live traffic cameras.")

mode_single, mode_grid, mode_route = st.tabs(["🔍 Single camera", "🗂️ Monitoring grid", "🛣️ Route"])

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
                    cam = filtered[choice]

                    fetch_col, weather_col, refresh_col = st.columns([1.3, 1.3, 1])
                    with fetch_col:
                        if st.button("Fetch live snapshot", type="primary"):
                            try:
                                st.session_state["live_frame"] = load_frame(cam["image_url"])
                                st.session_state["live_frame_stream_url"] = cam.get("stream_url", "")
                                st.session_state["live_frame_cam_name"] = cam["name"]
                                st.session_state["live_frame_cam_lat"] = cam.get("latitude")
                                st.session_state["live_frame_cam_lon"] = cam.get("longitude")
                                st.session_state["frame_source"] = "live"
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

                        @st.fragment(run_every=f"{refresh_secs}s", key="live_camera_auto_refresh")
                        def _auto_refresh_tick(cam=cam):
                            try:
                                st.session_state["live_frame"] = load_frame(cam["image_url"])
                                st.session_state["live_frame_stream_url"] = cam.get("stream_url", "")
                                st.session_state["live_frame_cam_name"] = cam["name"]
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

            now = datetime.now()
            try:
                predictions = _cached_predict_all(now.strftime("%Y-%m-%d %H:%M"), temp_c, rain_1h, snow_1h,
                                                   clouds_all, int(is_holiday), weather_main,
                                                   lag_1h, lag_3h, lag_24h)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.stop()

            cv_label = congestion_from_density(cv_result["density_ratio"])
            model_label = calibrated_label(predictions[model_choice])
            final_label = fuse_labels(model_label, cv_label)
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

            congestion_badge(final_label)
            if detect_anomaly(model_label, cv_label):
                st.warning("⚠️ **Unusual congestion detected** — the live camera reading differs sharply "
                          f"from what history expects for this time ({cv_label} vs. an expected "
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
            m1.metric("Vehicle count", cv_result["vehicle_count"])
            m2.metric("Image density ratio", cv_result["density_ratio"])
            st.caption(f"Live camera reading: **{cv_label}**  ·  "
                       f"{MODEL_LABELS[model_choice]} historical reading: **{model_label}**  ·  "
                       "final level takes whichever is more severe — a live camera showing gridlock "
                       "should never be reported as low congestion just because history says this hour is normally quiet.")

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
                    speed_limit_mph = st.number_input("Speed limit (mph)", min_value=5.0, value=65.0, step=5.0)

                if st.button("Check speeds"):
                    meters_per_pixel = lane_width_m / lane_width_px
                    with st.spinner("Capturing live video burst and tracking vehicles..."):
                        try:
                            st.session_state["speed_result"] = check_speeds(live_stream_url, meters_per_pixel, speed_limit_mph)
                            st.session_state["speed_result_cam_name"] = live_cam_name
                            st.session_state["speed_result_limit_mph"] = speed_limit_mph
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
                            veh_choice = st.selectbox("Vehicle", range(len(speeding_results)),
                                                      format_func=lambda i: veh_labels[i], key="echallan_vehicle")

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
                now = datetime.now()
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
                        cam_final = fuse_labels(hist_label, cam_cv_label) if hist_label else cam_cv_label
                        results.append({"camera": c, "result": cam_result, "cv_label": cam_cv_label,
                                        "final_label": cam_final, "error": None})
                    except Exception as e:
                        results.append({"camera": c, "result": None, "cv_label": None,
                                        "final_label": None, "error": str(e)})
                    progress.progress((i + 1) / len(selected_cams), text=f"Analyzed {i + 1}/{len(selected_cams)}")
                progress.empty()
                st.session_state["grid_results"] = results

            grid_results = st.session_state.get("grid_results")
            if grid_results:
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
with mode_route:
    st.subheader("Check congestion across multiple cameras on your route")
    st.caption("Pick a district, then select every camera along your route (any order) to get one "
               "overall reading — always driven by the worst segment, so a single jammed stretch is "
               "never hidden by an otherwise-clear route.")

    if not MODELS_READY:
        st.error(f"Trained model artifacts not found in {MODELS_DIR}. Run `python src/evaluate.py` first.")
    else:
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
            chosen_idx = st.multiselect("Cameras along your route (pick 2 or more)", range(len(route_cameras)),
                                        format_func=lambda i: route_labels[i], key="route_choice")

            with st.expander("Weather assumptions (applied to every camera on the route)"):
                rw1, rw2 = st.columns(2)
                with rw1:
                    rt_temp = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, key="route_temp")
                    rt_rain = st.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0, key="route_rain")
                with rw2:
                    rt_clouds = st.slider("Cloud cover (%)", 0, 100, 20, key="route_clouds")
                    rt_weather = st.selectbox("Weather", list(WEATHER_CODES.keys()), key="route_weather")

            if len(chosen_idx) < 2:
                st.info("Select at least 2 cameras to build a route.")
            elif st.button("Check route", type="primary"):
                selected_cams = [route_cameras[i] for i in chosen_idx]
                now = datetime.now()
                try:
                    model_pred = predict_all_models(now, rt_temp, rt_rain, 0.0, rt_clouds, 0, rt_weather)
                    hist_label = calibrated_label(model_pred["lstm"])
                except Exception as e:
                    st.error(f"Historical model prediction failed: {e}")
                    hist_label = None

                segment_results = []
                progress = st.progress(0.0, text="Checking route segments...")
                for i, c in enumerate(selected_cams):
                    try:
                        cam_frame = load_frame(c["image_url"])
                        cam_result = analyze_frame(cam_frame)
                        cam_cv_label = congestion_from_density(cam_result["density_ratio"])
                        cam_final = fuse_labels(hist_label, cam_cv_label) if hist_label else cam_cv_label
                        segment_results.append({"camera": c, "final_label": cam_final,
                                                "vehicle_count": cam_result["vehicle_count"], "error": None})
                    except Exception as e:
                        segment_results.append({"camera": c, "final_label": None,
                                                "vehicle_count": None, "error": str(e)})
                    progress.progress((i + 1) / len(selected_cams), text=f"Checked {i + 1}/{len(selected_cams)}")
                progress.empty()
                st.session_state["route_results"] = segment_results

            route_results = st.session_state.get("route_results")
            if route_results:
                valid = [r for r in route_results if r["final_label"]]
                if valid:
                    route_label = max((r["final_label"] for r in valid), key=CONGESTION_LEVELS.index)
                    worst_segment = next(r for r in valid if r["final_label"] == route_label)
                    st.markdown("###### Overall route congestion")
                    congestion_badge(route_label)
                    st.caption(f"Driven by the worst segment: **{worst_segment['camera']['name']}**")
                else:
                    st.warning("Couldn't read any segment on this route right now.")

                st.markdown("###### Segment-by-segment")
                for r in route_results:
                    seg_cols = st.columns([3, 1, 1])
                    seg_cols[0].markdown(f"**{r['camera']['name']}** — {r['camera']['nearby_place']}")
                    if r["error"]:
                        seg_cols[1].caption(f"⚠️ {r['error']}")
                    else:
                        seg_cols[1].markdown(f"{r['vehicle_count']} vehicles")
                        color = LEVEL_COLORS[r["final_label"]]
                        seg_cols[2].markdown(
                            f"<span style='color:{color};font-weight:700'>{r['final_label']}</span>",
                            unsafe_allow_html=True,
                        )
