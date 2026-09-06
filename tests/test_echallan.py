"""Tests for the simulated e-challan generator — pure logic, no live data needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from echallan import DISCLAIMER, compute_fine, generate_challan, render_challan_html  # noqa: E402


def _speeding_result(cls_name="car", speed_mph=80.0, over_mph=15.0):
    return {"cls_name": cls_name, "speed_mph": speed_mph, "speed_kmh": speed_mph * 1.60934,
            "speeding": True, "over_mph": over_mph, "last_box": (10, 10, 50, 50), "num_points": 3}


def test_compute_fine_base_slab_under_aggravated_threshold():
    fine, ref = compute_fine("car", over_mph=10.0)
    assert fine == 1000
    assert "183" in ref


def test_compute_fine_aggravated_slab_over_threshold():
    fine, _ = compute_fine("car", over_mph=25.0)
    assert fine == 2000


def test_compute_fine_heavier_vehicle_gets_higher_slab():
    car_fine, _ = compute_fine("car", over_mph=10.0)
    truck_fine, _ = compute_fine("truck", over_mph=10.0)
    assert truck_fine > car_fine


def test_compute_fine_unknown_class_falls_back_to_car_slab():
    unknown_fine, _ = compute_fine("spaceship", over_mph=10.0)
    car_fine, _ = compute_fine("car", over_mph=10.0)
    assert unknown_fine == car_fine


def test_generate_challan_raises_for_non_speeding_vehicle():
    not_speeding = _speeding_result()
    not_speeding["speeding"] = False
    with pytest.raises(ValueError):
        generate_challan(not_speeding, "Test Camera", speed_limit_mph=65.0)


def test_generate_challan_fields_populated_correctly():
    challan = generate_challan(_speeding_result(cls_name="truck", speed_mph=90.0, over_mph=25.0),
                                "I-5 Test Camera", speed_limit_mph=65.0, vehicle_number="KA01AB1234")
    assert challan.vehicle_class == "truck"
    assert challan.vehicle_number == "KA01AB1234"
    assert challan.speed_mph == 90.0
    assert challan.speed_limit_mph == 65.0
    assert challan.over_mph == 25.0
    assert challan.fine_inr == 4000  # truck, aggravated
    assert challan.challan_id.startswith("DEMO-")


def test_generate_challan_defaults_vehicle_number_placeholder_when_blank():
    challan = generate_challan(_speeding_result(), "Test Camera", speed_limit_mph=65.0)
    assert "NOT ENTERED" in challan.vehicle_number


def test_render_challan_html_includes_disclaimer_and_key_fields():
    challan = generate_challan(_speeding_result(cls_name="car", speed_mph=80.0, over_mph=15.0),
                                "Test Camera", speed_limit_mph=65.0, vehicle_number="XY12Z9999")
    html = render_challan_html(challan)
    assert "SIMULATED" in html
    assert DISCLAIMER in html
    assert "XY12Z9999" in html
    assert challan.challan_id in html
    assert "80.0" in html
