import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def plot_decision_tree():
    # Load the data used for training
    X = pd.read_csv("data/feature_importances.csv")
    model = DecisionTreeClassifier(random_state=42)

    # Ensure the model was trained with the data
    model.fit(X, pd.read_csv("data/predictions.csv").squeeze())

    # Create the decision tree plot
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
    plt.savefig("decision_tree_plot.png")  # Save the plot as an image
    plt.show()
    print("Decision tree plot saved as 'decision_tree_plot.png'.")

if __name__ == "__main__":
    plot_decision_tree()