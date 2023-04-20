from flask import Flask, render_template, request 
import pickle
app = Flask(__name__, static_folder='static')

#leemos
model = pickle.load(open("anuncios.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predecir", methods=['POST'])
def predecir():
    tiempo = int(request.form['tiempo'])
    prediction = model.predict([[tiempo]])
    output = round(prediction[0], 4)
    return render_template('index.html', prediction_text=f'Una persona que pase {tiempo} minutos en un sitio, tendra una probabilidad de {output} de darle click a un anuncio')

    
if __name__ == '__main__':
    app.run()
