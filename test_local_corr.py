import json
import importlib.util
import pathlib

# Dynamically load app.py since folder has a hyphen
app_path = pathlib.Path(__file__).resolve().parent / "app.py"
spec = importlib.util.spec_from_file_location("shap_app", app_path)
shap_app = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(shap_app)  # type: ignore

app = shap_app.app


def test_local_corr_endpoint():
    client = app.test_client()

    # Minimal dummy feature set (GeoJSON polygons)
    feature_template = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]
    }

    features = []
    for i in range(4):
        features.append({
            "geometry": feature_template,
            "attributes": {
                "varA": i + 1,
                "varB": (4 - i)
            }
        })

    payload = {
        "field_x": "varA",
        "field_y": "varB",
        "features": features,
        "k_neighbors": 2
    }

    response = client.post("/local_corr", data=json.dumps(payload), content_type="application/json")

    assert response.status_code in (200, 501), f"Unexpected status {response.status_code}"
    if response.status_code == 200:
        data = response.get_json()
        assert data.get("success") is True
        assert "features" in data
        assert len(data["features"]) == len(features)
    else:
        data = response.get_json()
        assert "error" in data 