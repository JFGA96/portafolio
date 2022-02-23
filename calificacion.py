from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

def data(text1,text2,text3,text4,text5):
   datos=[
      (text1,'Exelente'),
      (text2,'Muy bueno'),
      (text3,'Bueno'),
      (text4,'Regular'),
      (text5,'Mucho por mejorar'),
   ]
   return datos
#with open('datos.json','r') as fp:
   #print(type(fp))
def califi(datos,respuesta):
   cl=NaiveBayesClassifier(datos)
   print(cl.classify(respuesta))
   return cl.classify(respuesta)


#califi(complete,"Es un animal mamifero con sentidos muy desarrollados")

