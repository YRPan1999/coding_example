import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re

nltk.download("stopwords", quiet=True)  # Download stopwords quietly at the beginning

def read_article(file_name):
    try:
        with open(file_name, "r") as file:
            filedata = file.readlines()
            article = filedata[0].split(". ")
            sentences = []
    
            for sentence in article:
                # Using regex to clean the sentence
                cleaned_sentence = re.sub('[^a-zA-Z]', ' ', sentence)
                sentences.append(cleaned_sentence.split())
            if sentences:
                sentences.pop()  # Safely popping the last item if sentences list is not empty
            return sentences
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return []

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1 if w not in stopwords]
    sent2 = [w.lower() for w in sent2 if w not in stopwords]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    for w in sent1:
        vector1[all_words.index(w)] += 1
 
    for w in sent2:
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences = read_article(file_name)
    if not sentences:
        return "No content to summarize."

    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Use a dense numpy matrix to create the graph
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)  # Use the standard pagerank function

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    for i in range(min(top_n, len(ranked_sentence))):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text) + "."
