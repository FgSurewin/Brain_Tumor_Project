from re import X
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class train_model:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

    def train_random_forest_classifier(self):
        self.clf = RandomForestClassifier(n_estimators=300, random_state=0)
        self.clf.fit(self.X_train, self.y_train)
