from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# The score is the confidence level


# Here we use gpt2 to generate some text
generator = pipeline("text-generation", model="distilgpt2")
text_learning = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2
    )
text =  [x['generated_text'] for x in text_learning]
print(text)

# # we can also guess the most logical word, i.e. <mask>, in a sentence
# unmasker = pipeline("fill-mask")
# unmasker("In this course, we will teach you all about <mask> models.", top_k=2)

# # we can classify each word in a sentence; ner identifies entities such as persons, organizations or locations in a sentence
# net = pipeline("ner", grouped_entities=True)
# ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# # we can classify via labels
# classifier = pipeline("zero-shot-classification")
# res = classifier(text_learning,
#     # candidate_labels=["education", "politics", "business"] 

# More models available at: https://huggingface.co/models


# We specify parameters for our sentiment analysis model, e.g.,
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier(text)
print(res)

# # can rely on default values
# classifier = pipeline("sentiment-analysis")
# res = classifier(text)
# print(res)


# # We can save our tokens and model
# save_directory = "saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

# tok = AutoTokenizer.from_pretrained(save_directory)
# mod = AutoModForSequenceClassification.from_pretrained(save_directory)