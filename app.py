# web application by using flask

import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

f1=open('countvectorizer','rb')
cv=pickle.load(f1)


f2=open('lgmodel','rb')
model=pickle.load(f2)

f1.close()
f2.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['Review']
        data = [text]
        vectorizer = cv.transform(data).toarray()
        prediction = model.predict(vectorizer)
        if 'not' in text:
            prediction[0]=abs(prediction[0]-1)
        prediction=prediction[0]
    if prediction==1:
        return render_template('index.html', prediction_text='The review is Positive')
    else:
        return render_template('index.html', prediction_text='The review is Negative.')

if __name__ == "__main__":
    app.run(debug=True)
    
    
    