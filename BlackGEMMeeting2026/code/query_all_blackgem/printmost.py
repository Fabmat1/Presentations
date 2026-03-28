import pandas as pd

# Load the CSV file
df = pd.read_csv("output_full.csv")

# Get the 10 most frequent SOURCE_IDs
top_50 = df["SOURCE_ID"].value_counts().head(50)

# Print results
print(top_50)
