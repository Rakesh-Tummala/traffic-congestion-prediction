"""Streamlit dashboard: pick a real live Caltrans camera (click the map or use
the dropdown — they stay in sync), upload an image, or paste a snapshot URL;
set conditions; get vehicle detection plus a fused congestion prediction from
all three trained models. Also includes a monitoring grid to check several
cameras at once. Run with: streamlit run app.py"""
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
    forecast_day, fuse_labels, predict_all_models,
)
from speed_estimation import US_LANE_WIDTH_M, annotate_speeds, check_speeds  # noqa: E402

LEVEL_COLORS = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
MODEL_LABELS = {"linear_regression": "Linear Regression", "svr": "SVR", "lstm": "LSTM"}
MODEL_R2 = {"linear_regression": 0.948, "svr": 0.960, "lstm": 0.977}
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


def forecast_chart(df: pd.DataFrame, current_hour: int, bins: np.ndarray) -> alt.Chart:
    bands = pd.DataFrame({
        "y0": bins[:-1], "y1": bins[1:], "label": CONGESTION_LEVELS,
    })
    band_chart = alt.Chart(bands).mark_rect(opacity=0.12).encode(
        y="y0:Q", y2="y1:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=CONGESTION_LEVELS, range=list(LEVEL_COLORS.values())),
                         legend=alt.Legend(title="Congestion level")),
    )
    line = alt.Chart(df).mark_line(color="#e5e7eb", strokeWidth=2.5).encode(
        x=alt.X("hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23])),
        y=alt.Y("volume:Q", title="Predicted traffic volume"),
    )
    now_point = alt.Chart(df[df["hour"] == current_hour]).mark_point(
        size=160, color="white", filled=True, stroke="black", strokeWidth=2,
    ).encode(x="hour:Q", y="volume:Q", tooltip=["hour", "volume", "label"])
    return (band_chart + line + now_point).properties(height=280).interactive()


def camera_picker_map(filtered: list, selected_idx_key: str):
    """Render a clickable map of `filtered` cameras. Clicking a point selects
    that camera by writing into st.session_state[selected_idx_key] — the same
    session-state key the paired selectbox is bound to — so map clicks and
    dropdown choices stay in sync in either direction."""
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

    # The pickable layer uses ONLY static (non-per-row) color/radius. A per-row
    # accessor — either a column of list-valued cells, or separate r/g/b/a
    # columns referenced as get_fill_color=["r","g","b","a"] — silently breaks
    # picking (points still render, but clicks never register a selection),
    # confirmed by isolating this exact cause during development. The
    # currently-selected camera is instead highlighted with a second, separate
    # single-row layer drawn on top — visually distinct without touching the
    # main layer's accessors.
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
        pdk.Deck(layers=[base_layer, highlight_layer], initial_view_state=view,
                 map_style=None, tooltip={"text": "{name}"}),
        on_select="rerun", selection_mode="single-object", key=f"{selected_idx_key}_map",
    )
    st.caption("Click a point on the map to select that camera, or use the dropdown below.")

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

mode_single, mode_grid = st.tabs(["🔍 Single camera", "🗂️ Monitoring grid"])

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

                    if st.button("Fetch live snapshot", type="primary"):
                        try:
                            st.session_state["live_frame"] = load_frame(cam["image_url"])
                            st.session_state["live_frame_stream_url"] = cam.get("stream_url", "")
                            st.session_state["live_frame_cam_name"] = cam["name"]
                            st.session_state["frame_source"] = "live"
                        except Exception as e:
                            st.error(f"Could not load frame: {e}")

                    if st.session_state["frame_source"] == "live" and "live_frame" in st.session_state:
                        frame = st.session_state["live_frame"]
                        live_stream_url = st.session_state.get("live_frame_stream_url")
                        live_cam_name = st.session_state.get("live_frame_cam_name")
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

        st.subheader("2. Conditions")
        c1, c2 = st.columns(2)
        with c1:
            temp_c = st.slider("Temperature (°C)", -20.0, 45.0, 20.0)
            rain_1h = st.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0)
            snow_1h = st.number_input("Snow last hour (mm)", 0.0, 100.0, 0.0)
        with c2:
            clouds_all = st.slider("Cloud cover (%)", 0, 100, 20)
            weather_main = st.selectbox("Weather", list(WEATHER_CODES.keys()))
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
                    cv_result = analyze_frame(frame)
                except Exception as e:
                    st.error(f"Vehicle detection failed: {e}")
                    st.stop()

            annotated_rgb = cv2.cvtColor(cv_result["annotated_frame"], cv2.COLOR_BGR2RGB)

            now = datetime.now()
            try:
                predictions = predict_all_models(now, temp_c, rain_1h, snow_1h, clouds_all, int(is_holiday),
                                                  weather_main, lag_1h=lag_1h, lag_3h=lag_3h, lag_24h=lag_24h)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.stop()

            cv_label = congestion_from_density(cv_result["density_ratio"])
            model_label = calibrated_label(predictions[model_choice])
            final_label = fuse_labels(model_label, cv_label)

            congestion_badge(final_label)
            st.image(annotated_rgb, caption="Detected vehicles", use_container_width=True)

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

            st.markdown("##### Today's forecast")
            forecast_df = _cached_forecast(now.strftime("%Y-%m-%d"), temp_c, rain_1h, snow_1h,
                                            clouds_all, int(is_holiday), weather_main, model_choice)
            st.altair_chart(forecast_chart(forecast_df, now.hour, congestion_bins()), use_container_width=True)
            st.caption(f"Predicted volume across today using {MODEL_LABELS[model_choice]}, "
                       "weather held at the conditions set above. White dot marks the current hour.")

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
                            speed_result = check_speeds(live_stream_url, meters_per_pixel, speed_limit_mph)
                        except Exception as e:
                            st.error(f"Speed check failed: {e}")
                            speed_result = None

                    if speed_result is not None:
                        if not speed_result["results"]:
                            st.info("No vehicles could be tracked across the burst — try again "
                                    "(traffic is dynamic) or pick a busier camera.")
                        else:
                            annotated_speed_frame = annotate_speeds(speed_result["frames"][-1], speed_result["results"])
                            st.image(cv2.cvtColor(annotated_speed_frame, cv2.COLOR_BGR2RGB),
                                      caption=f"{live_cam_name} — tracked vehicle speeds", use_container_width=True)
                            speed_df = pd.DataFrame([
                                {"Vehicle": r["cls_name"], "Speed (mph)": round(r["speed_mph"], 1),
                                 "Speed (km/h)": round(r["speed_kmh"], 1),
                                 "Over limit (mph)": round(r["over_mph"], 1) if r["speeding"] else 0,
                                 "Speeding": "Yes" if r["speeding"] else "No"}
                                for r in sorted(speed_result["results"], key=lambda r: -r["speed_mph"])
                            ])
                            st.dataframe(speed_df, hide_index=True, use_container_width=True)
                            n_speeding = sum(r["speeding"] for r in speed_result["results"])
                            if n_speeding:
                                st.warning(f"{n_speeding} of {len(speed_result['results'])} tracked vehicle(s) "
                                           f"estimated over the {speed_limit_mph:.0f} mph limit.")

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
