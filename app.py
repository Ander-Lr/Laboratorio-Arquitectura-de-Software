from pathlib import Path
from flask import Flask, render_template, request, flash

from tuberias import TuberiaEntrenamiento
from servicios import ServicioPrediccion

# Configuración de Flask
app = Flask(
    __name__,
    static_folder="public",
    static_url_path="/public"
)

app.secret_key = "cambia-esta-clave-en-produccion"
MODEL_PATH = Path("models/weather_model.joblib")
TARGET = "rain"

tuberia = TuberiaEntrenamiento(TARGET, MODEL_PATH)
servicio_prediccion = ServicioPrediccion(MODEL_PATH)


def render_index(results=None, fields=None, prediction_result=None):
    if fields is None:
        fields = servicio_prediccion.campos()
    return render_template(
        "index.html",
        results=results,
        prediction_fields=fields,
        prediction_result=prediction_result
    )


@app.route("/", methods=["GET"])
def index():
    return render_index()

@app.route("/train", methods=["POST"])
def train():
    uploaded = request.files.get("dataset")
    if uploaded is None or not uploaded.filename:
        flash("Selecciona un archivo CSV.")
        return render_index()
    if not uploaded.filename.lower().endswith(".csv"):
        flash("El archivo debe tener extensión .csv.")
        return render_index()
    try:
        results = tuberia.run(uploaded.stream)
        return render_index(results=results)
    except Exception as exc:
        app.logger.exception("Error durante el entrenamiento")
        flash(f"No se pudo procesar el dataset: {exc}")
        return render_index()

@app.route("/predict", methods=["POST"])
def predict():
    if not servicio_prediccion.modelo_disponible():
        flash("Primero entrena un modelo subiendo un dataset.")
        return render_index()
    try:
        fields, prediction_result = servicio_prediccion.predecir(request.form)
        return render_index(fields=fields, prediction_result=prediction_result)
    except Exception as exc:
        app.logger.exception("Error durante la predicción")
        flash(f"No se pudo generar la predicción: {exc}")
        return render_index(fields=servicio_prediccion.campos(request.form))

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
