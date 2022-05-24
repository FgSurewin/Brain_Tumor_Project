import seaborn as sns
import matplotlib.pyplot as plt
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
    