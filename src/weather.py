"""Fetch real current weather for a location via Open-Meteo's free public API
(no API key needed) — lets the dashboard auto-fill its weather inputs from
the selected camera's actual coordinates instead of requiring manual entry."""
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes (what Open-Meteo returns) mapped onto this project's own
# weather categories (data_loader.WEATHER_CODES) — not a 1:1 mapping, just
# the closest reasonable bucket.
WMO_TO_WEATHER_MAIN = {
    0: "Clear",
    1: "Clouds", 2: "Clouds", 3: "Clouds",
    45: "Fog", 48: "Fog",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle", 56: "Drizzle", 57: "Drizzle",
    61: "Rain", 63: "Rain", 65: "Rain", 66: "Rain", 67: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    80: "Rain", 81: "Rain", 82: "Rain",
    85: "Snow", 86: "Snow",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


def fetch_current_weather(lat: float, lon: float, timeout: float = 8.0) -> dict:
    """Real current conditions at (lat, lon), in the same shape the
    dashboard's manual weather inputs use: {"temp_c", "rain_1h", "snow_1h",
    "clouds_all", "weather_main"} — ready to feed straight into
    build_feature_row / predict_all_models. Raises on a network error or an
    unexpected response shape; callers should catch and show it, the same
    way a failed camera fetch is already handled."""
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,rain,snowfall,cloud_cover,weather_code",
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    current = resp.json()["current"]
    return {
        "temp_c": float(current["temperature_2m"]),
        "rain_1h": float(current["rain"]),
        "snow_1h": float(current["snowfall"]) * 10,  # Open-Meteo reports snowfall in cm; dataset uses mm
        "clouds_all": float(current["cloud_cover"]),
        "weather_main": WMO_TO_WEATHER_MAIN.get(int(current["weather_code"]), "Clear"),
    }
