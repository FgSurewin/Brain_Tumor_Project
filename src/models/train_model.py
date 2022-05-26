from re import X
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd


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

    def train_majority_voting(self, lst_models):
        VC = VotingClassifier(estimators=lst_models, voting='soft')
        VC.fit(self.X_train, self.y_train)

    
    
class all_models:

    def __init__(self, X_train,X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test 

    
    def all_models(self):
    
        names = ["Naive_Bayes", "Nearest_Neighbors", "Logistic Regression", "Random_Forest", "RBF_SVM"]
        
        classifiers = [
            GaussianNB(),
            KNeighborsClassifier(n_neighbors=3),
            LogisticRegression(random_state=0, max_iter=1000, C=1.0),
            RandomForestClassifier(n_estimators=100 ,random_state=0),
            SVC(kernel="rbf", random_state=0)]


        classification_scores = []
        precision = []
        recall = []
        Fscore = []
        classificaation_auc = []


        for name, clf in zip(names, classifiers):
        
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            score = np.round(clf.score(self.X_test, self.y_test),2)
            prec = np.round(precision_score(self.y_test, y_pred, average='binary'),2)
            rec = np.round(recall_score(self.y_test, y_pred, average='binary'),2)
            fs= np.round(f1_score(self.y_test, y_pred, average='binary'),2)
            auc = np.round(roc_auc_score(self.y_test,y_pred, average = 'macro'),2)
            classification_scores.append(score)
            precision.append(prec)
            recall.append(rec)
            Fscore.append(fs)
            classificaation_auc.append(auc)

            
        result = pd.DataFrame()
        result['Classifier_name'] = names
        result['Accuracy'] = classification_scores
        result['Precision'] = precision
        result['Recall'] = recall
        result['F_Score'] = Fscore
        result['AUC'] = classificaation_auc
        
        
        return result
