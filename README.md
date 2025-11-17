# Medical Co-infection Prediction API

API để dự đoán đồng nhiễm vi khuẩn ở trẻ em bị nhiễm virus hô hấp.

## Models Available

1. **Clinical Only**: Chỉ dữ liệu lâm sàng (7 features)
2. **Clinical + Imaging**: Lâm sàng + hình ảnh học (11 features)  
3. **Clinical + Lab**: Lâm sàng + xét nghiệm (12 features)
4. **Full**: Đầy đủ tất cả (14 features)

## Installation
```bash
pip install -r requirements.txt
python main.py
```

## Usage
```python
import requests

# Test clinical only
data = {
    "temperature": 38.5,
    "age_months": 6.5, 
    "weight": 8.2,
    "spo2": 97,
    "height": 68,
    "respiratory_rate": 45,
    "heart_rate": 130
}

response = requests.post("http://localhost:8000/predict/clinical", json=data)
print(response.json())
```

## API Endpoints

- `GET /` - API info
- `GET /models` - Model information
- `POST /predict/clinical` - Clinical only prediction
- `POST /predict/clinical-imaging` - Clinical + imaging prediction
- `POST /predict/clinical-lab` - Clinical + lab prediction  
- `POST /predict/full` - Full prediction

## Model Performance

- Clinical Only: AUC = 0.596
- Clinical + Imaging: AUC = 0.801
- Clinical + Lab: AUC = 0.782
- Full Model: AUC = 0.888
