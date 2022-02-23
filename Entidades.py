import spacy

nlp = spacy.load("es_core_news_lg")
def identificacion(text):
    palabras=[]

    # Procesa el texto
    doc = nlp(text)
    print(text)

    # Itera sobre las entidades
    for ent in doc.ents:
        # Imprime en pantalla el texto de la entidad y su etiqueta
        print(ent.text, ent.label_)
        palabras.append(ent.text)
        palabras.append(ent.label_)
        #print("palabras"+palabras)
    return palabras
    # Obtén el span para "adidas zx"
    #adidas_zx = doc[14:16]

    # Imprime en pantalla el texto del span
    #print("Entidad faltante:", adidas_zx.text)