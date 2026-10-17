from datetime import datetime

ETIQUETAS = {
    "temperature": "Temperatura (°C)",
    "humidity": "Humedad (%)",
    "pressure": "Presión (hPa)",
    "wind_speed": "Velocidad del viento (km/h)"
}

CLASES = {
    "Yes": "Lloverá",
    "No": "No lloverá"
}

VARIANTES = {
    "Yes": {"modificador": "rain", "icono": "bi-cloud-rain-heavy-fill"},
    "No": {"modificador": "sun", "icono": "bi-sun-fill"}
}

VARIANTE_NEUTRA = {"modificador": "neutral", "icono": "bi-cloud-check-fill"}


def etiqueta(columna):
    return ETIQUETAS.get(str(columna), str(columna))


def clase(valor):
    return CLASES.get(str(valor), str(valor))


def pct(valor):
    return f"{float(valor):.1f}%"


def variante_prediccion(valor):
    variante = VARIANTES.get(str(valor), VARIANTE_NEUTRA)
    return {**variante, "texto": clase(valor)}


def fecha(valor):
    try:
        return datetime.fromisoformat(str(valor)).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return str(valor)


def registrar(app):
    app.add_template_filter(pct, "pct")
    app.add_template_filter(etiqueta, "etiqueta")
    app.add_template_filter(clase, "clase")
    app.add_template_filter(fecha, "fecha")
    app.add_template_global(variante_prediccion)
