import seaborn as sns
import matplotlib.pyplot as plt


class visualize:
    def __init__(self) -> None:
        pass

    def get_length_of_classes():
        sns.countplot(data=df, x=df['label'])
        plt.title('Length Of Classes')
        plt.show()
