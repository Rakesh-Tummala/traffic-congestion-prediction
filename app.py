"""Streamlit dashboard: upload/URL a camera frame, set conditions, get a fused
congestion prediction from the trained models. Run with: streamlit run app.py"""
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cv_module import analyze_frame, congestion_from_density  # noqa: E402
from data_loader import WEATHER_CODES  # noqa: E402
from live_cameras import DISTRICTS, fetch_cameras  # noqa: E402
from live_predict import build_feature_row, calibrated_label, fuse_labels, predict_volume  # noqa: E402


@st.cache_data(ttl=300)
def _cached_cameras(district: int):
    return fetch_cameras(district)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Smart Traffic Congestion Prediction", layout="wide")
st.title("Smart Traffic Congestion Prediction")
st.caption("Computer vision vehicle counting + Linear Regression / SVR, fused into a live congestion estimate.")

col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("1. Camera frame")
    tab_live, tab_upload, tab_url = st.tabs(["Live Caltrans camera", "Upload image", "Snapshot URL"])
    frame = None

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
            if search:
                needle = search.lower()
                cameras = [c for c in cameras if needle in c["name"].lower() or needle in c["nearby_place"].lower()]
            if cameras:
                labels = [f"{c['name']} — {c['nearby_place']}" for c in cameras]
                choice = st.selectbox(f"Camera ({len(cameras)} available)", range(len(cameras)),
                                      format_func=lambda i: labels[i])
                cam = cameras[choice]
                if st.button("Fetch live snapshot"):
                    from cv_module import load_frame
                    try:
                        frame = load_frame(cam["image_url"])
                        st.session_state["live_frame"] = frame
                    except Exception as e:
                        st.error(f"Could not load frame: {e}")
                elif "live_frame" in st.session_state:
                    frame = st.session_state["live_frame"]
            else:
                st.info("No cameras match that filter.")

    with tab_upload:
        uploaded = st.file_uploader("Traffic camera image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with tab_url:
        url = st.text_input("Live camera snapshot URL (public DOT traffic cam, etc.)")
        if url:
            from cv_module import load_frame
            try:
                frame = load_frame(url)
            except Exception as e:
                st.error(f"Could not load frame: {e}")

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

    model_choice = st.radio("Regression model", ["svr", "linear_regression"], horizontal=True)

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
    if frame is None:
        st.info("Upload an image or provide a snapshot URL to run a prediction.")
    else:
        with st.spinner("Running vehicle detection..."):
            cv_result = analyze_frame(frame)

        annotated_rgb = cv2.cvtColor(cv_result["annotated_frame"], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detected vehicles", use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Vehicle count", cv_result["vehicle_count"])
        m2.metric("Image density ratio", cv_result["density_ratio"])

        model_path = os.path.join(MODELS_DIR, f"{model_choice}.joblib")
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Run `python src/train_regression.py` first.")
        else:
            now = datetime.now()
            X = build_feature_row(now, temp_c, rain_1h, snow_1h, clouds_all, int(is_holiday), weather_main,
                                   lag_1h=lag_1h, lag_3h=lag_3h, lag_24h=lag_24h)
            predicted_volume = predict_volume(model_path, X)
            model_label = calibrated_label(predicted_volume)
            cv_label = congestion_from_density(cv_result["density_ratio"])
            final_label = fuse_labels(model_label, cv_label)

            st.metric("Congestion level", final_label)
            m3, m4 = st.columns(2)
            m3.metric(f"Historical model ({model_choice})", model_label, help=f"Predicted volume: {predicted_volume:.0f}")
            m4.metric("Live camera reading", cv_label, help=f"Density ratio: {cv_result['density_ratio']}")
            st.caption("Final level is the more severe of the two independent signals — "
                       "a live camera showing gridlock should never be reported as low congestion "
                       "just because history says this hour is normally quiet.")
