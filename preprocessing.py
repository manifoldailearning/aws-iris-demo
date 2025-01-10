import os
import pandas as pd

# Define input and output paths based on SageMaker Processing conventions
input_path = "/opt/ml/processing/input/iris.csv"
output_dir = "/opt/ml/processing/output"
output_path = os.path.join(output_dir, "processed_iris.csv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the Iris dataset
df = pd.read_csv(input_path)

# Identify numeric columns for normalization
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Apply normalization: (value - mean) / standard deviation
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std

# Save the processed data to the specified output path
df.to_csv(output_path, index=False)

print(f"Preprocessing complete. Processed file saved to {output_path}")
