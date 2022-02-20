import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer=WordNetLemmatizer()
stopwords=set(stopwords.words("english"))

print("Loading Model and Metadata...")
model=pickle.load(open("Topic_Recognizer.pkl","rb"))
topic_to_document=pickle.load(open("topic_to_document.pkl","rb"))
topics=pickle.load(open("Topics.pkl","rb"))

# print(topic_to_document)
text=input("Enter text: ")

def clean(text):
    words=text.lower().split(" ")
    cleaned_text=""
    for word in words:
        if word in stopwords: continue
        cleaned_text+=lemmatizer.lemmatize(word)+" "

    return cleaned_text

pred=np.array(model.predict([clean(text)])[0])
for ind in range(len(pred)):
    if pred[ind]>=0.25:
        print("Topic ->",topics[ind])
        print("Similar Documents ->",topic_to_document[ind])