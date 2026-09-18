"""Tests for the Open-Meteo real-weather fetch (network mocked — no live calls)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weather import WMO_TO_WEATHER_MAIN, fetch_current_weather  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_current_weather_maps_fields_correctly(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert params["latitude"] == 34.1
        assert params["longitude"] == -118.2
        return _FakeResponse({
            "current": {
                "temperature_2m": 22.5, "rain": 1.2, "snowfall": 0.5,
                "cloud_cover": 40, "weather_code": 61,
            }
        })

    monkeypatch.setattr("weather.requests.get", fake_get)
    result = fetch_current_weather(34.1, -118.2)
    assert result["temp_c"] == 22.5
    assert result["rain_1h"] == 1.2
    assert result["snow_1h"] == 5.0  # 0.5cm -> 5mm
    assert result["clouds_all"] == 40
    assert result["weather_main"] == "Rain"


def test_fetch_current_weather_unknown_wmo_code_falls_back_to_clear(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse({
            "current": {
                "temperature_2m": 15.0, "rain": 0.0, "snowfall": 0.0,
                "cloud_cover": 0, "weather_code": 9999,
            }
        })

    monkeypatch.setattr("weather.requests.get", fake_get)
    result = fetch_current_weather(0.0, 0.0)
    assert result["weather_main"] == "Clear"


def test_wmo_mapping_only_uses_known_project_weather_categories():
    from data_loader import WEATHER_CODES
    assert set(WMO_TO_WEATHER_MAIN.values()) <= set(WEATHER_CODES.keys())
