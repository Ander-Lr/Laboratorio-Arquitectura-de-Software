import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def prediction_fields(features, feature_types=None, values=None):
    feature_types = feature_types or {}
    values = values or {}
    return [
        {
            "name": feature,
            "numeric": str(feature_types.get(feature, "object")).lower()
            .startswith(("int", "uint", "float", "bool", "number")),
            "value": values.get(feature, "")
        }
        for feature in features
    ]


class ServicioPrediccion:
    def __init__(self, model_path):
        self.model_path = Path(model_path)

    def modelo_disponible(self):
        return self.model_path.exists()

    def campos(self, values=None):
        if not self.modelo_disponible():
            return []
        try:
            saved = joblib.load(self.model_path)
            return prediction_fields(
                saved.get("features", []),
                saved.get("feature_types", {}),
                values
            )
        except Exception:
            logger.exception("No se pudo cargar el esquema del modelo")
            return []

    def predecir(self, form):
        saved = joblib.load(self.model_path)
        features = saved["features"]
        model = saved["model"]
        feature_types = saved.get("feature_types", {})
        values = {}
        for feature in features:
            raw_value = form.get(feature, "").strip()
            if raw_value == "":
                raise ValueError(
                    f"El campo '{feature}' es obligatorio."
                )
            try:
                values[feature] = float(raw_value)
            except ValueError:
                values[feature] = raw_value
        sample = pd.DataFrame(
            [values],
            columns=features
        )
        prediction = model.predict(sample)[0]
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            probability = round(
                float(max(probabilities)) * 100,
                2
            )
        prediction_result = {
            "label": str(prediction),
            "probability": probability
        }
        return (
            prediction_fields(features, feature_types, values),
            prediction_result
        )
