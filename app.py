import joblib
import pandas as pd
import json
import os
from api.fraud.Fraud import Fraud
from flask import Flask, render_template, request

# loading model
model_path = os.path.join(os.getcwd(), 'models', 'model_cycle1.joblib')
model = joblib.load(model_path)

# initialize API
app = Flask(__name__)

transaction_type = ['CASH_IN',
         'CASH_OUT',
         'DEBIT',
         'PAYMENT',
         'TRANSFER']

@app.route('/')
def home():
    return render_template('index.html', transaction_type=sorted(transaction_type))

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        step = int(request.form['step'])
        type = request.form['transaction_type']
        amount = float(request.form['amount'])
        nameOrig = request.form['nameOrig']
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        nameDest = request.form['nameDest']
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
             
        input_df = pd.DataFrame({'step': [step], 'type': [type], 'amount': [amount],
                                 'nameOrig': [nameOrig], 'oldbalanceOrg': [oldbalanceOrg], 'newbalanceOrig': [newbalanceOrig],
                                 'nameDest': [nameDest], 'oldbalanceDest': [oldbalanceDest], 'newbalanceDest': [newbalanceDest]})
    
            
        # Instantiate Rossmann class
        pipeline = Fraud()
        
        # data cleaning
        df1 = pipeline.data_cleaning(input_df)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        prediction_json = pipeline.get_prediction(model, input_df, df3)
        prediction = json.loads(prediction_json)
        
        # Render result.html with prediction
        return render_template('result.html', prediction=prediction[0]['prediction'])

if __name__ == '__main__':
    app.run(debug=True)
