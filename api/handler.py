import joblib
import pandas as pd
from fraud.Fraud import Fraud
from flask import Flask, request, Response, render_template

# loading model
model = joblib.load('..\models\model_cycle1.joblib')

# initialize API
app = Flask(__name__)

transaction_type = ['CASH_IN',
         'CASH_OUT',
         'DEBIT',
         'PAYMENT',
         'TRANSFER']

steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

isFlaggedFraud = [0,1]

@app.route('/')
def home():
    return render_template('index.html', transaction_type=sorted(transaction_type), steps=sorted(steps), isFlaggedFraud=sorted(isFlaggedFraud))

@app.route('/predict', methods=['GET','POST'])
def churn_predict():
    
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
        isFlaggedFraud = request.form['isFlaggedFraud']

        input_df = pd.DataFrame({'step': [step], 'type': [type], 'amount': [amount],
                                 'nameOrig': [nameOrig], 'oldbalanceOrg': [oldbalanceOrg], 'newbalanceOrig': [newbalanceOrig],
                                 'nameDest': [nameDest], 'oldbalanceDest': [oldbalanceDest], 'newbalanceDest': [newbalanceDest], 'isFlaggedFraud': [isFlaggedFraud]})
    
            
        # Instantiate Rossmann class
        pipeline = Fraud()
        
        # data cleaning
        df1 = pipeline.data_cleaning(input_df)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        prediction = pipeline.get_prediction(model, input_df, df3)

        # Render result.html with prediction
        return render_template('result.html', prediction=prediction)
        
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run() 
