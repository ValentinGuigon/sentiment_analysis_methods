# pip install pandas, numpy, matplotlib, seaborn, nltk, transformers
# Do not forget to install Pytorch: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Also can benefit from installing Hugging Face xet (hf_xet): pip install huggingface_hub[hf_xet]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import nltk

nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

plt.style.use('ggplot')
df = pd.read_csv('./input/amazon-fine-food-reviews/Reviews.csv')
df = df.head(500)


## Quick EDA:
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()



## Basic NLTK:
example = df['Text'][50]
### demonstration of tokenizing mechanism
tokens = nltk.word_tokenize(example)
print(tokens[:10])
tagged = nltk.pos_tag(tokens) # categorize words as tags (e.g., singular noun, cardinal digit, etc.)
print(tagged[:10])
entities = nltk.chunk.ne_chunk(tagged) # chunks the list of tokens
print(entities)



## Step 1. VADER Sentiment Scoring: 
### "bag of words" approach: gets the neg/neutral/pos of the text, however doesn't account for relationship between words
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
# pip install jupyter | required
# pip install ipywidgets | required

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(example))

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


## Step 2. Roberta Pretrained Model:
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


sia = SentimentIntensityAnalyzer()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}') # It breaks because of the size of the data (tweet too long for the model)


## Step 3. Combine and compare the two models
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


## Step 4. Review examples

### False positive 1-Star Reviews
print(results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0])

print(results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0])

### False Negative 5-Star Reviews
print(results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0])

print(results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0])


## Extra: HuggingFace transformers pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier('I love sentiment analysis!')