from nltk.corpus import wordnet
import random

def make_keyword_recommendation(cluster_names):
    synonyms = []
    recommendations = []
    for cluster_name in cluster_names:
        for syn in wordnet.synsets(cluster_name):
            for l in syn.lemmas():
                synonyms.append(l.name())

        while cluster_name in synonyms:
            synonyms.remove(cluster_name)

        if synonyms:
            recommendations.append(synonyms[random.randint(0, len(synonyms)-1)])
        synonyms = []

    return recommendations


if __name__== "__main__":
    
    cluster_names = ["message", "voice", "chatbot"]
    
    print(make_keyword_recommendation(cluster_names))