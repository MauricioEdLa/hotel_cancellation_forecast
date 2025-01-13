import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def train_and_evaluate_decision_tree():
    # Step 1: Load data
    X = pd.read_csv("data/features.csv")
    y = pd.read_csv("data/target.csv")

    # Ensure y is a Series
    y = y.squeeze()

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Create and train the decision tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Make predictions
    y_pred = model.predict(X_test)

    # Step 5: Evaluate the model
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

    # Export predictions and actual values as CSV
    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    results.to_csv("data/predictions.csv", index=False)
    print("Predictions exported to 'predictions.csv'.")

    # Export feature importances as CSV
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importances.to_csv("data/feature_importances.csv", index=False)
    print("Feature importances exported to 'feature_importances.csv'.")

    # Visualize the decision tree and save the image
    plt.figure(figsize=(12,8))
    plot_tree(model, feature_names=X.columns, class_names=y.unique().astype(str), filled=True)
    plt.title("Decision Tree Visualization")
    plt.savefig("data/decision_tree_plot.png")
    print("Decision tree plot saved as 'decision_tree_plot.png'.")

if __name__ == "__main__":
    train_and_evaluate_decision_tree()