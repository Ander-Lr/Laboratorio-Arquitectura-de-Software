import joblib

class GuardarModeloFilter:
    def __init__(self, model_path):
        self.model_path = model_path

    def run(self, model, X, info, results):
        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "feature_types": {
                    column: str(dtype)
                    for column, dtype in X.dtypes.items()
                },
                "target": info["target"],
                "results": results
            },
            self.model_path
        )
        return self.model_path
