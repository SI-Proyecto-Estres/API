import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from model_loader import load_model_from_dir
from schemas import PredictionResponse, StressPrediction, TransformedMetric

DEFAULT_CONTRACT: Dict[str, Any] = {
    "modelo": "modelo_estres",
    "version": "1.0.0",
    "orden_entradas": [
        "sleep_hours_per_day",
        "study_hours_per_week",
        "social_support_score",
        "academic_pressure_score",
        "extracurricular_participation",
        "gpa_score_avg",
    ],
    "entradas": [
        {"nombre": "sleep_hours_per_day", "campo_formulario": "horas_sueno", "min": 4.0, "max": 8.0},
        {"nombre": "study_hours_per_week", "campo_formulario": "horas_estudio_semana", "min": 4.0, "max": 20.0},
        {"nombre": "social_support_score", "campo_formulario": "apoyo_social", "min": 0.0, "max": 10.0},
        {"nombre": "academic_pressure_score", "campo_formulario": "presion_academica", "min": 0.0, "max": 10.0},
        {"nombre": "extracurricular_participation", "campo_formulario": "extracurriculares", "min": 0.0, "max": 1.0},
        {"nombre": "gpa_score_avg", "campo_formulario": "promedio_0a10", "min": 0.0, "max": 100.0},
    ],
    "salida": {
        "clases": [
            {"id": 0, "etiqueta": "bajo", "mensaje": "Nivel de estrés bajo."},
            {"id": 1, "etiqueta": "medio", "mensaje": "Nivel de estrés moderado."},
            {"id": 2, "etiqueta": "alto", "mensaje": "Nivel de estrés alto."},
        ]
    },
    "explicacion": None,
    "aviso": "Herramienta de apoyo. No es un diagnóstico clínico. Entrenado con datos que no son de México.",
}

SOCIAL_SUPPORT_MAP = {
    "nada": 0.0,
    "poco": 3.3,
    "bastante": 6.7,
    "mucho": 10.0,
}


class StressService:
    def __init__(self, models_dir: str = "modelos", test_csv_path: str = "data/test.csv"):
        self.models_dir = Path(models_dir)
        self.test_csv_path = Path(test_csv_path)
        self.contract: Dict[str, Any] = DEFAULT_CONTRACT
        self.adapter = None
        self.reload()

    def reload(self) -> None:
        contract_path = self.models_dir / "contrato.json"
        if contract_path.exists():
            self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        else:
            self.contract = DEFAULT_CONTRACT
        self.adapter = load_model_from_dir(str(self.models_dir))

    def _parse_single_value(self, feature_name: str, raw_val: Any, from_form_alias: bool) -> float:
        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            raise ValueError(f"Falta un valor válido para '{feature_name}'.")

        if isinstance(raw_val, bool):
            val = 1.0 if raw_val else 0.0
        elif isinstance(raw_val, (int, float)):
            val = float(raw_val)
        else:
            text = str(raw_val).strip().lower()
            if feature_name == "social_support_score" and text in SOCIAL_SUPPORT_MAP:
                val = SOCIAL_SUPPORT_MAP[text]
            elif text in {"si", "sí", "yes", "true"}:
                val = 1.0
            elif text in {"no", "false"}:
                val = 0.0
            else:
                try:
                    val = float(text)
                except ValueError as exc:
                    raise ValueError(f"Valor inválido '{raw_val}' para '{feature_name}'.") from exc

        if feature_name == "gpa_score_avg" and (from_form_alias or val <= 10.0):
            val = val * 10.0

        return val

    def transform(
        self, features: Union[Dict[str, Any], List[float]]
    ) -> tuple[List[TransformedMetric], np.ndarray, List[TransformedMetric]]:
        entradas_meta = self.contract.get("entradas", DEFAULT_CONTRACT["entradas"])
        orden = self.contract.get("orden_entradas", [e["nombre"] for e in entradas_meta])
        meta_by_name = {e["nombre"]: e for e in entradas_meta}

        ordered_raw: List[Any] = []
        ordered_processed: List[float] = []

        if isinstance(features, list):
            if len(features) != len(orden):
                raise ValueError(f"Se esperaban {len(orden)} entradas, pero se recibieron {len(features)}.")
            for name, raw_val in zip(orden, features):
                spec = meta_by_name[name]
                val = float(np.clip(self._parse_single_value(name, raw_val, False), spec["min"], spec["max"]))
                ordered_raw.append(raw_val)
                ordered_processed.append(val)
        else:
            for name in orden:
                spec = meta_by_name[name]
                form_alias = spec.get("campo_formulario", name)
                if name in features:
                    raw_val, from_alias = features[name], False
                elif form_alias in features:
                    raw_val, from_alias = features[form_alias], True
                else:
                    raise ValueError(f"Falta la entrada requerida '{name}' o '{form_alias}'.")

                val = float(np.clip(self._parse_single_value(name, raw_val, from_alias), spec["min"], spec["max"]))
                ordered_raw.append(raw_val)
                ordered_processed.append(val)

        vector = np.array(ordered_processed, dtype=np.float32)

        exp = self.contract.get("explicacion") or {}
        media = np.asarray(exp.get("media", []), dtype=np.float32)
        desv = np.asarray(exp.get("desv", []), dtype=np.float32)
        direccion = np.asarray(exp.get("direccion", []), dtype=np.float32)
        textos = exp.get("textos", [])

        has_exp = media.size == vector.size and desv.size == vector.size and direccion.size == vector.size
        empujes: Optional[np.ndarray] = None
        top_indices: set[int] = set()

        if has_exp:
            safe_desv = np.where(desv == 0, 1.0, desv)
            empujes = direccion * (vector - media) / safe_desv
            candidates = [i for i, e in enumerate(empujes) if e > 0.5]
            candidates.sort(key=lambda i: float(empujes[i]), reverse=True)
            top_indices = set(candidates[:2])

        metrics: List[TransformedMetric] = []
        for idx, name in enumerate(orden):
            spec = meta_by_name[name]
            span = max(float(spec["max"]) - float(spec["min"]), 1e-6)
            norm_val = round(float((vector[idx] - float(spec["min"])) / span), 4)
            empuje_val = round(float(empujes[idx]), 4) if empujes is not None else None
            is_top = idx in top_indices

            if empuje_val is not None:
                status = "critico" if empuje_val > 1.0 else ("alerta" if empuje_val > 0.5 else "normal")
            else:
                status = "critico" if norm_val > 0.7 else ("alerta" if norm_val > 0.4 else "normal")

            metrics.append(
                TransformedMetric(
                    name=name,
                    form_field=spec.get("campo_formulario", name),
                    raw_value=ordered_raw[idx],
                    processed_value=round(float(vector[idx]), 4),
                    normalized_value=norm_val,
                    empuje=empuje_val,
                    is_top_factor=is_top,
                    factor_text=textos[idx] if idx < len(textos) else None,
                    status=status,
                )
            )

        top_factors = sorted(
            [m for m in metrics if m.is_top_factor],
            key=lambda m: (m.empuje if m.empuje is not None else 0.0),
            reverse=True,
        )
        return metrics, vector, top_factors

    def predict(
        self,
        features: Union[Dict[str, Any], List[float]],
        student_id: str | None = None,
        source: str = "live",
    ) -> PredictionResponse:
        transformed_metrics, vector, top_factors = self.transform(features)
        clases_meta = self.contract.get("salida", {}).get("clases", DEFAULT_CONTRACT["salida"]["clases"])
        labels = [c["etiqueta"] for c in clases_meta]
        messages = [c.get("mensaje", "") for c in clases_meta]

        if self.adapter is not None:
            probs = self.adapter.predict_proba(vector, num_classes=len(labels))
            model_used = f"{self.adapter.backend}:{self.adapter.source_path.name}"
        else:
            norm_arr = np.array([m.normalized_value for m in transformed_metrics], dtype=np.float32)
            risk_weights = np.array([-1.0, 0.7, -1.0, 1.2, -0.5, -0.8], dtype=np.float32)
            score = float(np.clip(0.5 + np.mean((norm_arr - 0.5) * risk_weights), 0.05, 0.95))
            raw_probs = np.array(
                [max(0.05, 1.0 - score * 1.3), max(0.1, 1.0 - abs(score - 0.5) * 1.6), max(0.05, (score - 0.35) * 1.4)],
                dtype=np.float32,
            )
            probs = raw_probs / raw_probs.sum()
            model_used = "fallback_heuristic"

        level = int(np.argmax(probs))
        prediction = StressPrediction(
            label=labels[level],
            level=level,
            message=messages[level],
            confidence=round(float(probs[level]), 4),
            probabilities={labels[i]: round(float(probs[i]), 4) for i in range(len(labels))},
        )

        return PredictionResponse(
            student_id=student_id,
            source=source,
            model_used=model_used,
            prediction=prediction,
            top_factors=top_factors,
            transformed_data=transformed_metrics,
            summary_vector=[round(float(x), 4) for x in vector],
            disclaimer=self.contract.get("aviso", DEFAULT_CONTRACT["aviso"]),
        )

    def get_test_samples(self, limit: int = 10) -> List[PredictionResponse]:
        orden = self.contract.get("orden_entradas", DEFAULT_CONTRACT["orden_entradas"])
        if self.test_csv_path.exists():
            df = pd.read_csv(self.test_csv_path)
            if "source_dataset" in df.columns:
                df = df[df["source_dataset"] == "stress_survey"]
            df = df.head(limit)
            samples: List[PredictionResponse] = []
            for idx, row in df.iterrows():
                row_dict = {col: row[col] for col in orden if col in df.columns}
                if len(row_dict) == len(orden):
                    samples.append(self.predict(row_dict, student_id=f"test_{idx}", source="test"))
            if samples:
                return samples

        demo_rows = [
            {
                "horas_sueno": 7.5,
                "horas_estudio_semana": 8.0,
                "apoyo_social": "mucho",
                "presion_academica": 3,
                "extracurriculares": "si",
                "promedio_0a10": 9.0,
            },
            {
                "horas_sueno": 6.0,
                "horas_estudio_semana": 12.0,
                "apoyo_social": "poco",
                "presion_academica": 6,
                "extracurriculares": "no",
                "promedio_0a10": 7.8,
            },
            {
                "horas_sueno": 4.5,
                "horas_estudio_semana": 18.0,
                "apoyo_social": "nada",
                "presion_academica": 9,
                "extracurriculares": "no",
                "promedio_0a10": 6.4,
            },
        ]
        return [self.predict(row, student_id=f"demo_{i}", source="test") for i, row in enumerate(demo_rows[:limit])]