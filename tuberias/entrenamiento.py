from datetime import datetime
from pathlib import Path

from filtros import (
    LoadDatasetFilter,
    ValidateDatasetFilter,
    PrepareDataFilter,
    DividirDatosFilter,
    TrainModelFilter,
    EvaluateModelFilter,
    GuardarModeloFilter
)

class TuberiaEntrenamiento:
    def __init__(self, target, model_path):
        self.target = target
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, archivo, nombre=None):
        raw = LoadDatasetFilter().run(archivo)
        validated = ValidateDatasetFilter(self.target).run(raw)
        X, y, info = PrepareDataFilter(self.target).run(validated)
        X_train, X_test, y_train, y_test = DividirDatosFilter().run(X, y)
        model = TrainModelFilter().run(X_train, y_train, info)
        metrics = EvaluateModelFilter().run(model, X_test, y_test)
        results = {
            "rows": len(validated),
            "features": len(X.columns),
            "target": info["target"],
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
            "metrics": metrics,
            "model_path": str(self.model_path),
            "dataset_name": nombre,
            "trained_at": datetime.now().isoformat(timespec="seconds")
        }
        GuardarModeloFilter(self.model_path).run(model, X, info, results)
        return results
