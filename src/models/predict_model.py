from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score


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

    def ten_fold_cross_val(self, models, model_labels):
        print('10-fold cross validation \n')
        for clf_, label in zip([models, model_labels]):
            scores = cross_val_score(estimator=clf_, X=self.X_train, y=self.y_train, cv=10, scoring='roc_auc')
            return scores.mean(), scores.std(), label
