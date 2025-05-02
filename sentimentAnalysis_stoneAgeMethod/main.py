import string
from collections import Counter
import matplotlib.pyplot as plt

# Cleaning Text Steps
# 1) Reading text file
text = open('./read.txt', encoding='utf-8').read()

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
tokenized_words = cleaned_text.split()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
              # stop words: commonly used words that do not add significant meaning or value

# 2) Removing stop words from the tokenized words list
final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)


# NLP Emotion Algorithm
# We use a mapping between words and emotions
# No training is involved
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

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()