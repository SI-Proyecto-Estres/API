from typing import List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from processor import StressService
from schemas import PredictRequest, PredictionResponse

app = FastAPI(
    title="Student Early Stress Detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = StressService(models_dir="modelos", test_csv_path="data/test.csv")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": service.adapter is not None,
        "model_backend": service.adapter.backend if service.adapter else None,
        "model_file": str(service.adapter.source_path.name) if service.adapter else None,
        "contract_loaded": (service.models_dir / "contrato.json").exists(),
        "expected_inputs": service.contract.get("orden_entradas"),
        "test_file_found": service.test_csv_path.exists(),
    }


@app.get("/contract")
def get_contract():
    return service.contract


@app.post("/reload-model")
def reload_model():
    service.reload()
    return {
        "reloaded": True,
        "model_loaded": service.adapter is not None,
        "model_backend": service.adapter.backend if service.adapter else None,
        "model_file": str(service.adapter.source_path.name) if service.adapter else None,
        "contract_loaded": (service.models_dir / "contrato.json").exists(),
    }


@app.get("/test-samples", response_model=List[PredictionResponse])
def get_test_samples(limit: int = Query(default=10, ge=1, le=100)):
    return service.get_test_samples(limit=limit)


@app.post("/predict", response_model=PredictionResponse)
def predict_stress(payload: PredictRequest):
    try:
        return service.predict(
            features=payload.features,
            student_id=payload.student_id,
            source="live",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno de inferencia: {exc}") from exc