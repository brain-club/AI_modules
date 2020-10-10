from pytorch_transformers import BertTokenizer, BertModel
from transformers import BartTokenizer, BartModel
from sentence_transformers import SentenceTransformer

import torch
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def class_embedding_from(sentence):
    '''
    #tokenizer = BartTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)

    hidden, cls_head = model(input_ids)

    # cls_head = [1, 768]

    embedding = cls_head.squeeze(0)

    # embedding = [768]
    '''
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embedding = model.encode(sentence)

    # embedding = [768] numpy array

    return embedding

def make_id2embedding(id2sentence):
    id2embedding = {}
    for id, sentence in id2sentence.items():
        print(f"embedding for sentence {id}")
        id2embedding[id] = class_embedding_from(sentence)
    
    # id2embeddings = {id: embedding}

    return id2embedding

def cluster_id2embedding(id2embedding):
    # embeddings = np.array([x.tolist() for x in id2embeddings.values()])
    embeddings = list(id2embedding.values())

    distance_threshold = 17.5
    cluster_numbers = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=distance_threshold).fit_predict(embeddings)

    id2cluster = {}
    for i, id in enumerate(id2embedding.keys()):
        id2cluster[id] = cluster_numbers[i]
    
    n_clusters = max(cluster_numbers) + 1

    return n_clusters, id2cluster

def display_sentence_clusters(id2sentence, id2cluster, n_clusters):
    sentences_cluster = {}
    for i in range(n_clusters):
        sentences_cluster[i] = []

    for id, sentence in id2sentence.items():
        cluster_id = id2cluster[id]
        sentences_cluster[cluster_id].append(sentence)

    for cluster_num, sentences in sentences_cluster.items():
        print(f"\nCluster {cluster_num}:\n{sentences}\n")

def organizer(id2sentence):
    '''
        INPUT:
            id2sentence : (dict) with sentence id as key and sentence as its value
        OUTPUT:
            n_clusters : (int) number of clusters
            id2cluster : (dict) with sentence id as key and assigned cluster numebr as its value
    '''
    id2embedding = make_id2embedding(id2sentence)
    n_clusters, id2cluster = cluster_id2embedding(id2embedding)

    return n_clusters, id2cluster


if __name__== "__main__":
    
    id2sentence = {
        0: "automatically mute the voice when you don't intended",
        1: "Chatbot service - where newly joined company member can get the answers via text",
        2: "Chatbot for freshman in universities can ask anything freely via text",
        3: "Loction-based service: the members' location are shown on the map and they can interact with other members via message",
        4: "Meet your neighbor - SNS service"
    }
    
    n_clusters, id2cluster = organizer(id2sentence)

    print(f"\n- {n_clusters} clusters are generated.")
    display_sentence_clusters(id2sentence, id2cluster, n_clusters)

