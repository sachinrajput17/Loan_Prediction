import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    labels = ["Not Eligible for Loan","Eligible for Loan"]
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    final_features = pd.DataFrame(final_features, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 
                                       'Credit_History', 'Property_Area'],
                                       dtype=float)
    prediction = model.predict(final_features)

    output = labels[prediction[0]]

    return render_template('index.html', prediction_text='You are  {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
