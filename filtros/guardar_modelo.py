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
                "feature_ranges": {
                    column: {
                        "min": float(X[column].min()),
                        "max": float(X[column].max())
                    }
                    for column in info["numeric_columns"]
                    if X[column].notna().any()
                },
                "target": info["target"],
                "results": results
            },
            self.model_path
        )
        return self.model_path
