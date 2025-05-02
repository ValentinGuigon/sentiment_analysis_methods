# pip install nltk
import nltk
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Cleaning Text Steps
# 1) Reading text file
text = open('read.txt', encoding='utf-8').read()

# 2) Convert to lowercase
lower_case = text.lower()

# 3) Remove punctuations
# str1: specifies the list of characters that need to be replaced
# str2: specifies the list of characters with which the characters need to be replaced
# str3: specifies the list of characters that needs to be deleted
# -> we don't need to replaced but only need to delete, therefore we only specify str3
# Returns: returns the translation table which specifies the conversions that can be used
cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))


# Tokenization
# 1) Splitting text into words
tokenized_words = word_tokenize(cleaned_text, "english")

# 2) Removing stop words from the tokenized words list
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)


# NLP Emotion Algorithm
emotion_list = []
# 1) Check if the word in the final word list is also present in emotion.txt
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

# 2) If word is present, add the emotion to emotion_list
        if word in final_words:
            emotion_list.append(emotion)

# 3) Finally count each emotion in the emotion_list
w = Counter(emotion_list)
print(w)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score ['neg']
    pos = score ['pos']
    neu = score ['neu']
    if neg > pos:
        print("Negative Sentiment")
    elif pos > neg:
        print("Positive Sentiment")
    else:
        print("Neutral Vibe")
        

sentiment_analyse(cleaned_text)

# plotting the emotions on the graph
fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()