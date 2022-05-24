from re import X
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class train_model:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

    def train_random_forest_classifier(self):
        self.clf = RandomForestClassifier(n_estimators=300, random_state=0)
        self.clf.fit(self.X_train, self.y_train)

    def train_SVC(self):
        SVM = SVC(kernel='rbf', probability=True)
        SVM.fit(self.X_train, self.y_train)

    def train_KNN(self):
        neigh = KNeighborsClassifier()
        neigh.fit(self.X_train, self.y_train)

    def train_gaussian_nb(self):
        gnb = GaussianNB()
        gnb.fit(self.X_train, self.y_train)

    def train_logistic_regression(self):
        lr = LogisticRegression(random_state=0)
        lr.fit(self.X_train, self.y_train)

    def train_linear_svc(self):
        linear_SVM = SVC(kernel='linear', probability=True)
        linear_SVM.fit(self.X_train, self.y_train)
