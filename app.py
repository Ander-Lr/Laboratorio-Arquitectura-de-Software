from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request, flash
from sklearn.model_selection import train_test_split

from filtros import (
    LoadDatasetFilter,
    ValidateDatasetFilter,
    PrepareDataFilter,
    TrainModelFilter,
    EvaluateModelFilter
)

# Configuración de Flask
app = Flask(
    __name__,
    static_folder="public",
    static_url_path="/public"
)

app.secret_key = "cambia-esta-clave-en-produccion"
MODEL_PATH = Path("models/weather_model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TARGET = "rain"


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


def saved_prediction_fields():
    if not MODEL_PATH.exists():
        return []
    try:
        saved = joblib.load(MODEL_PATH)
        return prediction_fields(
            saved.get("features", []),
            saved.get("feature_types", {})
        )
    except Exception:
        app.logger.exception("No se pudo cargar el esquema del modelo")
        return []


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        results=None,
        prediction_fields=saved_prediction_fields(),
        prediction_result=None
    )

@app.route("/train", methods=["POST"])
def train():
    uploaded = request.files.get("dataset")
    if uploaded is None or not uploaded.filename:
        flash("Selecciona un archivo CSV.")
        return index()
    if not uploaded.filename.lower().endswith(".csv"):
        flash("El archivo debe tener extensión .csv.")
        return index()
    try:
        raw = LoadDatasetFilter().run(uploaded.stream)
        validated = ValidateDatasetFilter(TARGET).run(raw)
        X, y, info = PrepareDataFilter(TARGET).run(validated)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
        model = TrainModelFilter().run(
            X_train,
            y_train,
            info
        )
        metrics = EvaluateModelFilter().run(
            model,
            X_test,
            y_test
        )
        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "feature_types": {
                    column: str(dtype)
                    for column, dtype in X.dtypes.items()
                },
                "target": info["target"]
            },
            MODEL_PATH
        )
        results = {
            "rows": len(validated),
            "features": len(X.columns),
            "target": info["target"],
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
            "metrics": metrics,
            "model_path": str(MODEL_PATH)
        }
        return render_template(
            "index.html",
            results=results,
            prediction_fields=prediction_fields(
                X.columns,
                {column: str(dtype) for column, dtype in X.dtypes.items()}
            ),
            prediction_result=None
        )
    except Exception as exc:
        app.logger.exception("Error durante el entrenamiento")
        flash(f"No se pudo procesar el dataset: {exc}")
        return index()

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists():
        flash("Primero entrena un modelo subiendo un dataset.")
        return index()
    saved = {}
    try:
        saved = joblib.load(MODEL_PATH)
        features = saved["features"]
        model = saved["model"]
        feature_types = saved.get("feature_types", {})
        values = {}
        for feature in features:
            raw_value = request.form.get(feature, "").strip()
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

        return render_template(
            "index.html",
            results=None,
            prediction_fields=prediction_fields(
                features,
                feature_types,
                values
            ),
            prediction_result=prediction_result
        )

    except Exception as exc:
        app.logger.exception("Error durante la predicción")
        flash(f"No se pudo generar la predicción: {exc}")
        return render_template(
            "index.html",
            results=None,
            prediction_fields=prediction_fields(
                saved.get("features", []),
                saved.get("feature_types", {}),
                request.form
            ),
            prediction_result=None
        )

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
