import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk

plt.style.use('ggplot')
df = pd.read_csv('./input/amazon-fine-food-reviews/Reviews.csv')
df = df.head(500)


## Basic NLTK:
example = df['Text'][50]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens) # categorize words as tags (e.g., singular noun, cardinal digit, etc.)
entities = nltk.chunk.ne_chunk(tagged) # chunks the list of tokens


## Step 1. VADER Sentiment Scoring: 
### "bag of words" approach: gets the neg/neutral/pos of the text, however doesn't account for relationship between words
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
# pip install jupyter | required
# pip install ipywidgets | required

sia = SentimentIntensityAnalyzer()

#### Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T #store in a pandas dataframe
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

#### Plot VADER results: we look at whether lower scores get negative compound scores, etc.
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Stars Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

#### This confirms that VADER does what it should do