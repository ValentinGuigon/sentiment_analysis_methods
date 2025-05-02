import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm.notebook import tqdm


plt.style.use('ggplot')
df = pd.read_csv('./input/amazon-fine-food-reviews/Reviews.csv')
df = df.head(500)
example = df['Text'][50]


## Roberta Pretrained Model:
### Deep learning model that can pickup on data context; we use HuggingFace
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
# pip install PyTorch | required
# e.g.: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Downloads and uses pre-trained model, trained on twitter
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Run for Roberta Model
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt') # pt for pyTorch tensors
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy() # transform tensor into numpy
    scores = softmax(scores)
    scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
    }
    return scores_dict


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        roberta_result = polarity_scores_roberta(text)
        res[myid] = roberta_result
    except RuntimeError:
        print(f'Broke for id {myid}') # It breaks because of the size of the data (tweet too long for the model)


## Combine and compare the two models
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

sns.pairplot(data=results_df,
             vars=['roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


## Review examples
### False positive 1-Star Reviews : # We check for text said to be positive by model but that has a score of 1 given by the user
print(results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]) 

### False Negative 5-Star Reviews
print(results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0])