from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import nltk
from nltk.corpus import stopwords
import string
import json

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)
processed_texts=[]

def center():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    texts_list=[]
    with open('') as fewrel_val_1600_output:
        data=json.load(fewrel_val_1600_output)
        text_list=[]
        for i in range(len(data)):
            text_list.append(data[i]["output"])
            if i % 100 == 99:
                texts_list.append(text_list[:])
                text_list.clear()
    stop_words = set(stopwords.words('english'))

    for list in texts_list:
        processed_texts.append([preprocess_text(text) for text in list])

    text_embeddings = []
    text_embeddings_all = []
    for text_list in processed_texts:
        text_embeddings.clear()
        for text in text_list:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                outputs = model(input_ids)
            text_embedding = torch.mean(outputs[0], dim=1).numpy()
            text_embeddings.append(text_embedding)
        text_embeddings_all.append(text_embeddings[:])

    center_label_list=[]
    for i in range(len(text_embeddings_all)):
        text_embeddings=text_embeddings_all[i]
        similarity_matrix = 1 - pairwise_distances(np.concatenate(text_embeddings, axis=0), metric="cosine")

        overall_center_label_index = np.argmax([similarity_matrix[i][j] for i in range(len(texts_list[i])) for j in range(len(texts_list[i]))])
        overall_center_label = texts_list[i][overall_center_label_index]
        center_label_list.append(overall_center_label)

    with open("") as center_1:
        json.dump(center_label_list,center_1)

