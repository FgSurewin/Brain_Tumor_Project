

class predict_model:
    def __init__(self, model, X_test, y_test) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    # Return y_test_pred
    def predict_model(self):
        return self.predict(self.X_test)
