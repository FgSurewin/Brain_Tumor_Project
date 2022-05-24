from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class predict_model:
    def __init__(self, model, X_test, y_test) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    # Return y_test_pred
    def predict_model(self):
        return self.predict(self.X_test)

    def get_classification_report(self):
        return metrics.classification_report(self.y_test, self.predict_model())

    def generate_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.predict_model())
        cm_display = ConfusionMatrixDisplay(cm).plot()
