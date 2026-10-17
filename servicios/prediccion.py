import logging

import pandas as pd

from .validacion import (
    LIMITES_FISICOS,
    LONGITUD_MAXIMA_TEXTO,
    ErrorValidacion,
    validar_numero,
    validar_texto
)

logger = logging.getLogger(__name__)


def es_numerico(tipo):
    return str(tipo).lower().startswith(
        ("int", "uint", "float", "bool", "number")
    )


def prediction_fields(features, feature_types=None, values=None,
                      feature_ranges=None, errors=None, warnings=None):
    feature_types = feature_types or {}
    values = values or {}
    feature_ranges = feature_ranges or {}
    errors = errors or {}
    warnings = warnings or {}
    fields = []
    for feature in features:
        numeric = es_numerico(feature_types.get(feature, "object"))
        minimo, maximo = LIMITES_FISICOS.get(feature, (None, None))
        fields.append({
            "name": feature,
            "numeric": numeric,
            "value": values.get(feature, ""),
            "min": minimo if numeric else None,
            "max": maximo if numeric else None,
            "maxlength": None if numeric else LONGITUD_MAXIMA_TEXTO,
            "range": feature_ranges.get(feature) if numeric else None,
            "error": errors.get(feature),
            "warning": warnings.get(feature)
        })
    return fields


class ServicioPrediccion:
    def __init__(self, repositorio):
        self.repositorio = repositorio

    def modelo_disponible(self):
        return self.repositorio.existe()

    def campos(self, values=None):
        if not self.modelo_disponible():
            return []
        try:
            saved = self.repositorio.cargar()
            return prediction_fields(
                saved.get("features", []),
                saved.get("feature_types", {}),
                values,
                saved.get("feature_ranges", {})
            )
        except Exception:
            logger.exception("No se pudo cargar el esquema del modelo")
            return []

    def predecir(self, form):
        saved = self.repositorio.cargar()
        features = saved["features"]
        model = saved["model"]
        feature_types = saved.get("feature_types", {})
        feature_ranges = saved.get("feature_ranges", {})
        raw_values, values, errors, warnings = {}, {}, {}, {}
        for feature in features:
            raw_value = form.get(feature, "").strip()
            raw_values[feature] = raw_value
            if raw_value == "":
                errors[feature] = "Este campo es obligatorio."
                continue
            if es_numerico(feature_types.get(feature, "object")):
                valor, error, aviso = validar_numero(
                    feature, raw_value, feature_ranges.get(feature)
                )
            else:
                valor, error, aviso = validar_texto(raw_value)
            if error:
                errors[feature] = error
                continue
            values[feature] = valor
            if aviso:
                warnings[feature] = aviso

        if errors:
            # Se conserva lo que escribió el usuario para que pueda corregirlo
            raise ErrorValidacion(prediction_fields(
                features, feature_types,
                {**raw_values, **values},
                feature_ranges, errors, warnings
            ))

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
            "probability": probability,
            "extrapolated": bool(warnings)
        }
        return (
            prediction_fields(
                features, feature_types, values,
                feature_ranges, warnings=warnings
            ),
            prediction_result
        )
