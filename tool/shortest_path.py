import spacy
from spacy import displacy
from spacy.tokens import Token
import networkx as nx
from stanfordcorenlp import StanfordCoreNLP
import networkx as nx

def shortest_dependency_path(doc, word1, word2):
    edges = []
    for token in doc:
        edges.append((token.head.i, token.i))
    graph = nx.Graph(edges)
    path = nx.shortest_path(graph, source=word1.i, target=word2.i)
    return [doc[i].text for i in path]


def run(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    displacy.serve(doc, style='dep')
    word1 = doc[4]  
    word2 = doc[11]  
    path_words = shortest_dependency_path(doc, word1, word2)


def get_shortest_path(sentence,word1_index,word2_index):
    dependencies = nlp.dependency_parse(sentence)
    G = nx.Graph()
    for dep in dependencies:
        relation, governor, dependent = dep
        if governor != 0:  
            G.add_edge(governor, dependent, label=relation)
    res=[]
    try:
        shortest_path = nx.shortest_path(G, source=word1_index, target=word2_index)
        for idx in shortest_path:
            curr=nlp.word_tokenize(sentence)[idx - 1]
            res.append(curr)
    except nx.NetworkXNoPath:     
    return res