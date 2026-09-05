"""Streamlit dashboard: pick a real live Caltrans camera (or upload an image /
paste a snapshot URL), set conditions, and get vehicle detection plus a fused
congestion prediction from all three trained models. Run with: streamlit run app.py"""
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

from cv_module import analyze_frame, congestion_from_density  # noqa: E402
from data_loader import WEATHER_CODES  # noqa: E402
from live_cameras import DISTRICTS, fetch_cameras  # noqa: E402
from live_predict import (  # noqa: E402
    CONGESTION_LEVELS, MODELS_DIR, build_feature_row, calibrated_label, congestion_bins,
    forecast_day, fuse_labels, predict_all_models,
)

LEVEL_COLORS = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
MODEL_LABELS = {"linear_regression": "Linear Regression", "svr": "SVR", "lstm": "LSTM"}
MODEL_R2 = {"linear_regression": 0.948, "svr": 0.960, "lstm": 0.977}


@st.cache_data(ttl=300)
def _cached_cameras(district: int):
    return fetch_cameras(district)


@st.cache_data(ttl=1800)
def _cached_forecast(date_str: str, temp_c: float, rain_1h: float, snow_1h: float,
                      clouds_all: float, is_holiday: int, weather_main: str, model: str):
    base = datetime.strptime(date_str, "%Y-%m-%d")
    return forecast_day(base, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main, model)


def congestion_badge(level: str):
    color = LEVEL_COLORS[level]
    st.markdown(
        f"""<div style="background:{color}1a;border:2px solid {color};border-radius:14px;
        padding:18px 24px;text-align:center;margin-bottom:12px;">
        <div style="font-size:13px;color:{color};font-weight:700;letter-spacing:2px;
        text-transform:uppercase;">Congestion Level</div>
        <div style="font-size:44px;color:{color};font-weight:800;line-height:1.2;">{level}</div>
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


st.set_page_config(page_title="Smart Traffic Congestion Prediction", layout="wide", page_icon="🚦")
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Traffic Congestion Prediction")
st.caption("Computer vision vehicle counting + Linear Regression / SVR / LSTM, fused with real live traffic cameras.")

col_input, col_result = st.columns([1, 1.2], gap="large")

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
            filtered = cameras
            if search:
                needle = search.lower()
                filtered = [c for c in cameras if needle in c["name"].lower() or needle in c["nearby_place"].lower()]

            if filtered:
                labels = [f"{c['name']} — {c['nearby_place']}" for c in filtered]
                choice = st.selectbox(f"Camera ({len(filtered)} available)", range(len(filtered)),
                                      format_func=lambda i: labels[i])
                cam = filtered[choice]

                map_df = pd.DataFrame([
                    {"lat": float(c["latitude"]), "lon": float(c["longitude"]),
                     "name": c["name"], "selected": c is cam}
                    for c in filtered if c["latitude"] and c["longitude"]
                ])
                if not map_df.empty:
                    layer = pdk.Layer(
                        "ScatterplotLayer", data=map_df,
                        get_position="[lon, lat]",
                        get_fill_color="selected ? [239, 68, 68, 230] : [100, 116, 139, 140]",
                        get_radius="selected ? 400 : 200",
                        pickable=True,
                    )
                    view = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=8)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                              map_style=None, tooltip={"text": "{name}"}))

                if st.button("Fetch live snapshot", type="primary"):
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
    models_ready = all(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in ("linear_regression.joblib", "svr.joblib", "lstm.pt", "lstm_scalers.joblib", "seasonal_profile.joblib")
    )
    if not models_ready:
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
