from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
import re  
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


with open('notebook/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


with open('notebook/vectorizer.pkl', 'rb') as vec_file:
    countvec = pickle.load(vec_file)


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get('sentence')
    print(f"Sentence received: {sentence}")
    
    # preprocessing the sentence
    sen = re.sub("[^a-zA-Z]", " ", sentence)
    sen = sen.lower()
    sen = countvec.transform([sen])
    preds = model.predict(sen)
    print(f"Preprocessing completed: {preds}")
    
    if preds == 0:
        result = "yes"
    else:
        result = "no"
    
    print(f"sending result: {result}")
    return jsonify({ 'prediction': result })


if __name__ == '__main__':
    app.run()