
import requests
import json

# API endpoint (thay đổi URL khi deploy)
BASE_URL = "http://localhost:8000"

# Test data
test_clinical = {
    "temperature": 38.5,
    "age_months": 6.5,
    "weight": 8.2,
    "spo2": 97,
    "height": 68,
    "respiratory_rate": 45,
    "heart_rate": 130
}

test_clinical_imaging = {
    **test_clinical,
    "infiltration": "1 = có",
    "lesion_location": "2 bên", 
    "air_trapping": "1 = có",
    "bronchial_thickening": "2 = Không"
}

test_clinical_lab = {
    **test_clinical,
    "crp": 25.5,
    "wbc": 15000,
    "lymphocyte_percent": 35.0,
    "neutrophil_percent": 60,
    "hemoglobin": 9.8
}

test_full = {
    **test_clinical,
    "infiltration": "1 = có",
    "lesion_location": "2 bên",
    "air_trapping": "1 = có", 
    "crp": 25.5,
    "wbc": 15000,
    "lymphocyte_percent": 35.0,
    "neutrophil_percent": 60
}

def test_endpoint(endpoint, data):
    url = f"{BASE_URL}/predict/{endpoint}"
    response = requests.post(url, json=data)
    print(f"\n=== Test {endpoint.upper()} ===")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']}")
        print(f"Model: {result['model_used']}")
        print(f"AUC: {result['model_auc']}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    # Test all endpoints
    test_endpoint("clinical", test_clinical)
    test_endpoint("clinical-imaging", test_clinical_imaging)
    test_endpoint("clinical-lab", test_clinical_lab)
    test_endpoint("full", test_full)
