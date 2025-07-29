import nltk
import pandas as pd
from nltk.classify import NaiveBayesClassifier

nltk.download('punkt_tab')
nltk.download('stopwords')

train_data = [
    ('Finance describes the flow of money and how people, companies, and governments make...raising capital and issuing stock...', 'finance'),
    ("Primary health care enables health systems...health promotion to disease prevention...", 'healthcare'),
    ("For entrepreneurs, students, creators...the phenomenal performance of M4 comes to Mac...", 'tech')
]
def word_features(text):
    words = nltk.word_tokenize(text.lower())
    return {word: True for word in words if word.isalpha()}

featuresets = [ (word_features(text), label) for (text, label) in train_data ]

classifier = NaiveBayesClassifier.train(featuresets)
text=input("Enter text to classify: ")
label = classifier.classify(word_features(text))
print(f"The text is classified as: {label}")