# sentiment_analysis_methods
Various methods for sentiment analysis. From everything manual - no supervised modelling involved, to using distilbert

To get data, execute the following:
```python
import kagglehub
import shutil
import os
import pandas as pd

# Download to cache
cache_path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")

# Create ./input/ folder if it doesn't exist
local_path = "./input/amazon-fine-food-reviews"
os.makedirs(local_path, exist_ok=True)

# Copy all files to ./input/
for file_name in os.listdir(cache_path):
    shutil.copy(os.path.join(cache_path, file_name), local_path)
```
