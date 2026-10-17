from sklearn.model_selection import train_test_split

class DividirDatosFilter:
    def __init__(self, test_size=0.20, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def run(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
