import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker automatically sets these environment variables and/or passes arguments
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))

    args = parser.parse_args()

    # Construct the path to the preprocessed Iris dataset
    train_data_path = os.path.join(args.train, "processed_iris.csv")

    # Load the preprocessed data
    print(f"Loading training data from {train_data_path}")
    df = pd.read_csv(train_data_path)

    # Separate features and target
    X = df.drop("species", axis=1)
    y = df["species"]

    # Initialize and train the model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model to the specified directory
    model_output_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_output_path)
    print(f"Model trained and saved to {model_output_path}")
