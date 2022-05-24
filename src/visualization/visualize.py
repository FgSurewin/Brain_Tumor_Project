import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc
import numpy as np


class visualize:
    def __init__(self, pca, pca_reduced, df) -> None:
        self.pca = pca
        self.pca_reduced = pca_reduced
        self.df = df

    def get_length_of_classes(self):
        sns.countplot(data=self.df, x=self.df['label'])
        plt.title('Length Of Classes')
        plt.show()

    def get_explained_variance(self):
        plt.grid()
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_*100))
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.show()

    def view_pca_with_n_components(self):
        pca_recovered = self.pca.inverse_transform(self.pca_reduced)
        img_pca_10 = pca_recovered[1, :].reshape([256, 256])
        plt.imshow(img_pca_10, cmap='gray_r')
        plt.title('Compressed image with n components', fontsize=15, pad=15)
        plt.show()

    def create_ROC(self, all_clf, clf_labels, colors, linestyles, X_test, y_test):
        figure(figsize=(10, 8), dpi=80)
        for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):

            y_pred = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_true=y_test.values.astype(int), y_score=y_pred)

            roc_auc = auc(x=fpr, y=tpr)

            plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1],
                 linestyle='--',
                 color='gray',
                 linewidth=2)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid(alpha=0.5)
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.show()
