from fastapi import APIRouter, HTTPException, Request
from app.schemas.proxy_schema import PredictRequest, PredictResponse
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import traceback

# ========== PATH ==========
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "ml" / "models"
# Load artifacts
model = joblib.load(MODEL_DIR / "model_lgbm.joblib")
label_encoders = joblib.load(MODEL_DIR / "label_encoders_lgbm.joblib")
feature_columns = joblib.load(MODEL_DIR / "features.joblib")

router = APIRouter(prefix="/proxy", tags=["Proxy Detection"])


# ========= PREDICT =========
@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    try:
        new_df = pd.DataFrame([req.dict()], copy=False)

        # ========= ENCODING =========
        for col, le in label_encoders.items():
            if col in new_df:
                new_df[col] = new_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # ========= DEFAULT VALUES =========
        new_df["latitude"] = new_df.get("latitude", 0.0)
        new_df["longitude"] = new_df.get("longitude", 0.0)

        # ASN fix
        if "asn" in new_df.columns:
            new_df["asn"] = (
                new_df["asn"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .fillna("0")
                .astype("int32")
            )

        # ========= ALIGN FEATURES =========
        for col in feature_columns:
            if col not in new_df:
                new_df[col] = 0

        new_df = new_df[feature_columns]

        # ========= PREDICTION =========
        prediction = int(model.predict(new_df)[0])
        confidence = model.predict_proba(new_df)[0][prediction]
        confidence_percent = round(confidence * 100, 2)

        return PredictResponse(
            prediction_code=prediction,
            detectionMethod=["LightGBM"],
            confidence=confidence_percent,
            is_proxy=bool(prediction),
        )

    except Exception as err:
        logging.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail="Prediction Failed"
        )