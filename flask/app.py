from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open(r'C:\Users\DELL\OneDrive\Desktop\Online payments fraud detection\flask\payments.pkl', 'rb'))

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Predict page
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Submit route
@app.route('/submit', methods=['POST'])
def submit():
    try:
        step = float(request.form['step'])
        type_ = float(request.form['type'])
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        features = np.array([[step, type_, amount, oldbalanceOrg,
                              newbalanceOrig, oldbalanceDest, newbalanceDest]])

        prediction = model.predict(features)
        

        return render_template('submit.html', prediction)
    

    except Exception as e:
        return str(e)

if __name__=="__main__":
    app.run(debug=True)