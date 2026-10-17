import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class RepositorioModelo:
    """Fuente única del estado del modelo entrenado (artefacto en disco).

    Mantiene en memoria la última versión leída y solo vuelve a cargar el
    archivo cuando cambia, por ejemplo tras un reentrenamiento.
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._cache = None
        self._version = None

    def existe(self):
        return self.model_path.exists()

    def cargar(self):
        stat = self.model_path.stat()
        version = (stat.st_mtime_ns, stat.st_size)
        if self._version != version:
            self._cache = joblib.load(self.model_path)
            self._version = version
        return self._cache

    def resultados(self):
        if not self.existe():
            return None
        try:
            return self.cargar().get("results")
        except Exception:
            logger.exception("No se pudieron leer los resultados del modelo")
            return None
