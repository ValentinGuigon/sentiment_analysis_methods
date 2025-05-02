# sentiment_analysis_methods
Various methods for sentiment analysis. From everything manual - no supervised modelling involved, to using distilbert

To get data, execute the following:
```python
import kagglehub
import pandas as pd

# Download the dataset
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")

# Load the CSV file
csv_path = f"{path}/Reviews.csv"
df = pd.read_csv(csv_path)

# Display basic info
print("Dataset loaded.")
print(df.head())
```
