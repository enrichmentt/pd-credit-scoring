"""FastAPI-сервис PD-модели.

Endpoints:
    GET  /health  — проверка, что сервис и модель живы;
    POST /predict — скоринг одного клиента (класс + вероятность дефолта).

Запуск: uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.models.predict import load_model, predict_proba

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield


app = FastAPI(
    title="PD Credit Scoring API",
    description="Скоринговая модель вероятности дефолта (UCI Credit Card)",
    version="1.0.0",
    lifespan=lifespan,
)


class CreditApplication(BaseModel):
    """Сырые признаки клиента (как в датасете UCI)."""

    LIMIT_BAL: float = Field(..., ge=0, description="Кредитный лимит")
    SEX: int = Field(..., ge=1, le=2, description="1 = муж., 2 = жен.")
    EDUCATION: int = Field(..., ge=0, le=6, description="1-4 основные категории")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1 = женат/замужем, 2 = холост")
    AGE: int = Field(..., ge=18, le=100)
    PAY_0: int = Field(..., ge=-2, le=9)
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


class Prediction(BaseModel):
    default: int = Field(..., description="1 = дефолт, 0 = нет")
    probability: float = Field(..., description="Вероятность дефолта")
    threshold: float = 0.5


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok", "model_path": str(config.MODEL_PATH)}


@app.post("/predict", response_model=Prediction)
def predict(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    raw = pd.DataFrame([application.model_dump()])
    proba = float(predict_proba(model, raw).iloc[0])
    return Prediction(default=int(proba >= 0.5), probability=round(proba, 6))
