from bs4 import BeautifulSoup
import requests
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def busqueda(palabra):
    filtered_word=[]
    lista_sinonimos=[]
    sinonimo=str(palabra)
    #print(sinonimo)
    url="https://www.wordreference.com/sinonimos/"
    buscar= url+sinonimo
    resp=requests.get(buscar)
    soup=BeautifulSoup(resp.text)
    lista=soup.find(class_='trans clickable')
    #if lista==None:
     #   pass
    #else:
    try:
        sino=lista.find('li')
        lista_sinonimos.append(sino.next_element)
        #print(lista_sinonimos)

        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(str(lista_sinonimos))
        text = ' '.join(word for word in text)
        tokenized_word=word_tokenize(text)
        tokenized_word = [word.lower() for word in tokenized_word]

        stop_words = set(stopwords.words('Spanish'))
        all_stopwords = stopwords.words('Spanish')
        all_stopwords.append('si')
        all_stopwords.append('min')
        all_stopwords.append('tan') 

        filtered_word = []
        for word in tokenized_word:
            if word not in all_stopwords:
                filtered_word.append(word)
        #print(filtered_word[0])
        return filtered_word
    except:
        error="error"
        return error

#print(busqueda("buscar"))