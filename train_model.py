import os
import nltk
import xml.etree.ElementTree as ET
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ktrain
import pickle
import numpy as np
from collections import defaultdict

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer=WordNetLemmatizer()
stopwords=set(stopwords.words("english"))

def to_string_utf8(document):
    return document.decode('utf-8')

def get_doc_data(filepath):
    tree = ET.parse(filepath)
    document = ET.tostring(tree.getroot(), encoding='utf-8', method='text')
    document = to_string_utf8(document)
    document = re.sub('[ \t\n]+', ' ', document)
    return document

def clean(text):
    words=text.lower().split(" ")
    cleaned_text=""
    for word in words:
        if word in stopwords: continue
        cleaned_text+=lemmatizer.lemmatize(word)+" "

    return cleaned_text

documents=[]
path="data/"
for doc in os.listdir(path):
    if doc.endswith(".xml"):
        try:
            documents.append([doc,clean(get_doc_data(os.path.join(path, doc)))])
        except: pass

documents=np.array(documents)
model=ktrain.text.get_topic_model(documents[:,1])
model.build(documents[:,1], threshold=0.25)
model.print_topics(show_counts=True)

file=open("Topic_Recognizer.pkl","wb")
pickle.dump(model,file)
print("MODEL SAVED")

print("Saving Metdata...")
topic_to_document=defaultdict(list)
topics=model.get_topics()
for i in documents:
    pred=model.predict([i[1]])[0]
    for ind in range(len(pred)):
        if pred[ind]>=0.25:
            topic_to_document[ind].append(i[0])

file=open("topics.pkl","wb")
pickle.dump(topics, file)
print("TOPICS SAVED")

file=open("topic_to_document.pkl", "wb")
pickle.dump(topic_to_document, file)
print("MAPPER CREATED")