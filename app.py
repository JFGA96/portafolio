from flask import Flask,jsonify,request
from Comparacion_texto import notas
from Entidades import identificacion
from Keywords import ejecucionkey
from calificacion import califi,data
from Sinonimos import busqueda

app= Flask(__name__)

@app.route('/comparacion', methods=['POST'])
def comparaciondetexto():
    texto1=request.json['texto1']
    texto2=request.json['texto2']
    return jsonify({"respuesta":notas(texto1,texto2)})

@app.route('/entidades', methods=['POST'])
def entidades():
    text=request.json['text']
    return jsonify({"respuesta":identificacion(text)})

@app.route('/palabrasclave', methods=['POST'])
def words():
    text=request.json['text']
    cant=request.json['cant']
    return jsonify({"respuesta":ejecucionkey(text,cant)})

@app.route('/sinonimos', methods=['POST'])
def similares():
    palabra=request.json['text']
    return jsonify({"respuesta":busqueda(palabra)})

@app.route('/calificaciones', methods=['POST'])
def evaluar():
    complete=data("Mamífero carnívoro doméstico de la familia de los cánidos que se caracteriza por tener los sentidos del olfato y el oído muy finos, por su inteligencia y por su fidelidad al ser humano, que lo ha domesticado desde tiempos prehistóricos; hay muchísimas razas, de características muy diversas.","Mamífero doméstico de la familia de los cánidos, de tamaño, forma y pelaje muy diversos, según las razas, que tiene olfato muy fino y es inteligente y muy leal a su dueño"
    ,"un perro es un animal peludo con colmillos y domestico","un perro es un animal con garras y con muchos poderes","los perros son animales callejeros y con enfermedades")
    respuesta = request.json['resp']
    return jsonify({"calificacion":califi(complete,respuesta)})

if __name__ == '__main__':
    app.run(debug=True , port=4000)