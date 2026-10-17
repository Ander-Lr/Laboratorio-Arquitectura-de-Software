import math

# Límites físicos: fuera de ellos el valor es imposible y se rechaza
LIMITES_FISICOS = {
    "temperature": (-60.0, 60.0),
    "humidity": (0.0, 100.0),
    "pressure": (870.0, 1085.0),
    "wind_speed": (0.0, 400.0)
}

LONGITUD_MAXIMA_TEXTO = 100


class ErrorValidacion(ValueError):
    def __init__(self, campos):
        self.campos = campos
        total = sum(1 for campo in campos if campo["error"])
        super().__init__(
            "Revisa el campo marcado." if total == 1
            else f"Revisa los {total} campos marcados."
        )


def numero(valor):
    return f"{valor:g}"


def validar_numero(feature, raw, rango=None):
    """Devuelve (valor, error, aviso) para un campo numérico."""
    try:
        valor = float(raw.replace(",", "."))
    except ValueError:
        return None, "Ingresa un número válido.", None
    if not math.isfinite(valor):
        return None, "Ingresa un número finito.", None

    minimo, maximo = LIMITES_FISICOS.get(feature, (None, None))
    if minimo is not None and valor < minimo:
        return None, f"El valor mínimo permitido es {numero(minimo)}.", None
    if maximo is not None and valor > maximo:
        return None, f"El valor máximo permitido es {numero(maximo)}.", None

    aviso = None
    if rango and not rango["min"] <= valor <= rango["max"]:
        aviso = (
            f"Fuera del rango de entrenamiento ({numero(rango['min'])} – "
            f"{numero(rango['max'])}); la predicción puede ser menos confiable."
        )
    return valor, None, aviso


def validar_texto(raw):
    if len(raw) > LONGITUD_MAXIMA_TEXTO:
        return None, f"Máximo {LONGITUD_MAXIMA_TEXTO} caracteres.", None
    return raw, None, None
