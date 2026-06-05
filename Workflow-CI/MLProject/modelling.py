import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # Set MLflow experiment
    mlflow.set_experiment("Wine_Quality_CI")

    # Enable Autolog
    mlflow.sklearn.autolog()

    # Load Data
    df = pd.read_csv('/content/wine_clean.csv')
    X = df.drop(columns=['quality'])
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model within an MLflow Run
    with mlflow.start_run(run_name="CI_Model"):
        print("Starting CI model training...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("CI model training complete! Metrics logged automatically by Autolog.")
