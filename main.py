
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="Medical Co-infection Prediction API", version="1.0.0")

# Load models và metadata
model_info = joblib.load('models/model_info.pkl')
feature_mapping = joblib.load('models/feature_mapping.pkl')

# Load trained models
models = {}
for model_key in ['clinical', 'clinical_imaging', 'clinical_lab', 'full']:
    models[model_key] = joblib.load(f'models/{model_key}_model.pkl')

# Pydantic models cho input
class ClinicalInput(BaseModel):
    temperature: float  # Nhiệt độ (°C)
    age_months: float  # Tuổi (tháng)
    weight: float  # Cân nặng (kg)
    spo2: int  # SpO2 (%)
    height: int  # Chiều cao (cm)
    respiratory_rate: int  # Nhịp thở (lần/phút)
    heart_rate: int  # Mạch (lần/phút)

class ClinicalImagingInput(ClinicalInput):
    infiltration: str  # "1 = có" hoặc "2 = Không"
    lesion_location: str  # "1 bên" hoặc "2 bên"
    air_trapping: str  # "1 = có" hoặc "2 = Không"
    bronchial_thickening: str  # "1 = có" hoặc "2 = Không"

class ClinicalLabInput(ClinicalInput):
    crp: float  # CRP (mg/L)
    wbc: float  # WBC (/mm³)
    lymphocyte_percent: float  # L (%)
    neutrophil_percent: int  # N (%)
    hemoglobin: float  # Hemoglobin (g/dL)

class FullInput(BaseModel):
    # Clinical
    temperature: float
    age_months: float
    weight: float
    spo2: int
    height: int
    respiratory_rate: int
    heart_rate: int
    # Imaging
    infiltration: str
    lesion_location: str
    air_trapping: str
    # Lab
    crp: float
    wbc: float
    lymphocyte_percent: float
    neutrophil_percent: int

def predict_coinfection(model_key: str, input_data: dict) -> dict:
    """Dự đoán đồng nhiễm"""
    try:
        # Load model package
        model_package = models[model_key]
        model = model_package['model']
        scaler = model_package['scaler']
        imputer = model_package['imputer']
        label_encoders = model_package['label_encoders']
        
        # Tạo dataframe từ input
        feature_map = feature_mapping[model_key]
        df_input = pd.DataFrame([input_data])
        
        # Map tên features
        df_mapped = pd.DataFrame()
        for api_name, original_name in feature_map.items():
            if api_name in df_input.columns:
                df_mapped[original_name] = df_input[api_name]
        
        # Encode categorical variables
        for col in df_mapped.columns:
            if col in label_encoders:
                try:
                    df_mapped[col] = label_encoders[col].transform(df_mapped[col].astype(str))
                except:
                    # Nếu giá trị mới chưa thấy, dùng giá trị phổ biến nhất
                    df_mapped[col] = 0
        
        # Impute missing values
        X_imputed = imputer.transform(df_mapped)
        
        # Scale features
        X_scaled = scaler.transform(X_imputed)
        
        # Predict
        probability = model.predict_proba(X_scaled)[0][1]
        prediction = "Có đồng nhiễm" if probability >= 0.5 else "Không đồng nhiễm"
        confidence = "Cao" if abs(probability - 0.5) >= 0.3 else "Trung bình" if abs(probability - 0.5) >= 0.15 else "Thấp"
        
        return {
            "model_used": model_info[model_key]['name'],
            "prediction": prediction,
            "probability": round(probability, 3),
            "confidence_level": confidence,
            "model_auc": round(model_info[model_key]['auc'], 3),
            "recommendation": "Cần theo dõi và điều trị kháng sinh" if probability >= 0.7 
                            else "Cân nhắc điều trị kháng sinh" if probability >= 0.5
                            else "Có thể không cần kháng sinh, theo dõi thêm"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/")
def root():
    return {"message": "Medical Co-infection Prediction API", "version": "1.0.0"}

@app.get("/models")
def get_models():
    return model_info

@app.post("/predict/clinical")
def predict_clinical(data: ClinicalInput):
    return predict_coinfection('clinical', data.dict())

@app.post("/predict/clinical-imaging") 
def predict_clinical_imaging(data: ClinicalImagingInput):
    return predict_coinfection('clinical_imaging', data.dict())

@app.post("/predict/clinical-lab")
def predict_clinical_lab(data: ClinicalLabInput):
    return predict_coinfection('clinical_lab', data.dict())

@app.post("/predict/full")
def predict_full(data: FullInput):
    return predict_coinfection('full', data.dict())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
