
###Mejorar para añadir varios documentos y compararlos


import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


nlp = spacy.load("es_core_news_lg")

def libreria_spacy(contexto,texto):
    
    notas_spacy=[]
    doc2 = (str(contexto))
    doc3=nlp(doc2)
    doc = nlp(str(texto))
    var= doc.similarity(doc3)
    notas_spacy.append(var)
    #print(notas_spacy) 
    return notas_spacy

def libreria_sklearn(contexto, texto):
    notas_sklearn=[]
    notas_sklearn.append(contexto)
    notas_sklearn.append(texto)
    vectorizer = CountVectorizer()
    x=vectorizer.fit_transform(notas_sklearn)
    tfidf = TfidfVectorizer().fit_transform(notas_sklearn)
    notas=[cosine_similarity(tfidf[0:1],tfidf[1:2])]

    return notas

def notas(texto1,texto2):
    #n1=libreria_sklearn(texto1,texto2)
    n2=libreria_spacy(texto1,texto2)
    #print(n1[0][0][0],n2[0])
    #n3=(n1[0][0][0]+n2[0])/2
    porcentaje=n2[0]*100
    similaridad=(round(porcentaje,2))
    return similaridad
    
#print(notas("a mi me gusta la comida","a mi me gustan los deportes"))