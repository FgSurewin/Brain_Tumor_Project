import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import rank
from numpy.random import randint

import matplotlib.colors as colors
colors_list = list(colors._colors_full_map.values())

colors_dark = ["#404040", "#676c72", '#737980', '#8e959e', '#f9f9f9']
colors_red = ["#EA4335", "#E57373", '#EF9A9A', '#FFCDD2', '#FFEBEE']
colors_green = ['#34A853','#81C784','#A5D6A7','#C8E6C9','#E8F5E9']
colors_blue = ['#4285F4','#42A5F5','#90CAF9','#BBDEFB','#E3F2FD']
colors_yellow = ['#FBBC04','#FFCA28','#FFE082','#FFECB3','#FFF8E1']

class visualize:
    def __init__(self, pca, pca_reduced, df,img, mask) -> None:
        self.pca = pca
        self.pca_reduced = pca_reduced
        self.df = df
        self.img = img
        self.mask =mask

    def get_length_of_classes(self):
        sns.countplot(data=self.df, x=self.df['label'])
        plt.title('Length Of Classes')
        plt.show()
    
    def Display_img_mask2(self):
        gray_img = rgb2gray(self.img)

        fig, axes = plt.subplots(1, 3, figsize=(10, 8))
        ax = axes.ravel()
        ax[0].imshow(self.img)
        ax[0].set_title("Original")
        ax[1].imshow(gray_img, cmap=plt.cm.gray)
        ax[1].set_title("Grayscale")
        ax[2].imshow(self.mask)
        ax[2].set_title("Mask")
        fig.tight_layout()



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

    def plot_loss_and_acurracy(self, history):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
        epochs = range(len(history.history['loss']))

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        axes[0].plot(epochs, loss, 'y', label='Training loss')
        axes[0].plot(epochs, val_loss, 'r', label='Validation loss')
        axes[0].set_title('Training and validation loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']
        axes[1].plot(epochs, acc, 'y', label='Training acc')
        axes[1].plot(epochs, val_acc, 'r', label='Validation acc')
        axes[1].set_title('Training and validation accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

    def plot_history_precision_recall(self, history):
        epochs = range(len(history.history['loss']))
        fig, ax = plt.subplots(1,2,figsize=(14,7))
        train_acc = history.history['precision_2']
        train_loss = history.history['recall_2']
        val_acc = history.history['val_precision_2']
        val_loss = history.history['val_recall_2']

        fig.text(s='Epochs vs. Training and Validation Precision/Recall',size=18,fontweight='bold',
                    fontname='monospace',color=colors_dark[0],y=1,x=0.28,alpha=0.8)

        sns.despine()
        ax[0].plot(epochs, train_acc, color=colors_green[1],
                label = 'Training Precision')
        ax[0].plot(epochs, val_acc, color=colors_yellow[1],
                label = 'Validation Precision')
        ax[0].legend(frameon=False)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Training & Validation Precision')

        sns.despine()
        ax[1].plot(epochs, train_loss, color=colors_green[1],
                label ='Training Recall')
        ax[1].plot(epochs, val_loss, color=colors_yellow[1],
                label = 'Validation Recall')
        ax[1].legend(frameon=False)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Training & Validation Recall')

        fig.show()

    def plot_roc_curve(models):
        auc_list = []

        plt.figure(figsize=(12, 8))
        plt.plot([0, 1], [0, 1], 'y--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')

        for idx, (name, model) in enumerate(models):
            y_preds = model.predict(X_test).ravel()
            fpr, tpr, thresholds = roc_curve(y_test, y_preds) 
            auc_value = auc(fpr, tpr)
            auc_list.append(auc_value)
            label = str(name) + " - AUC: " + str(round(auc_value, 3))
            plt.plot(fpr, tpr, marker='.', color=colors_list[randint(len(colors_list) - 1)], label=label)

        plt.legend(fontsize=28)
        
        
        return auc_list

