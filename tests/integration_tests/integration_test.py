import pytest
import requests


def get_request(api_path, payload):
    r = requests.get(f"http://localhost:9070/{api_path}", params=payload)
    return r


def test_response_recommendations_no_metadata():
    api_path = "recommendations"
    payload_no_metadata = {"user_id": 18}

    response = get_request(api_path, payload_no_metadata)

    # Assertion:
    assert response.status_code == 200  # Validation of status code
    data = response.json()

    # Assertion of body response content:
    assert isinstance(data, dict)
    assert isinstance(data["items"], list) and len(data["items"]) > 0
    assert all(isinstance(sub, dict) for sub in data["items"])
    assert all(isinstance(sub["id"], str) for sub in data["items"])


def test_response_recommendations_with_metadata():
    api_path = "recommendations"
    payload_with_metadata = {"user_id": 18, "returnMetadata": True}

    response = get_request(api_path, payload_with_metadata)

    # Assertion:
    assert response.status_code == 200
    data = response.json()
    
    # Assertion of body response content:
    assert isinstance(data, dict)
    assert isinstance(data["items"], list) and len(data["items"]) > 0

    assert all(isinstance(sub, dict) for sub in data["items"])
    assert all(isinstance(sub["id"], str) for sub in data["items"])
    assert all(isinstance(sub["title"], str) for sub in data["items"])
    assert all(isinstance(sub["genres"], list) for sub in data["items"])


def test_response_features():
    api_path = "features"
    payload_with_metadata = {"user_id": 18}

    response = get_request(api_path, payload_with_metadata)

    # Assertion:
    assert response.status_code == 200
    data = response.json()

    # Assertion of body response content:
    assert isinstance(data, dict)
    assert isinstance(data["features"], list) and len(data["features"]) == 1
    assert all(isinstance(sub, dict) for sub in data["features"])
    assert all(isinstance(sub["histories"], list) for sub in data["features"])
    assert all(len(sub["histories"]) > 0 for sub in data["features"])