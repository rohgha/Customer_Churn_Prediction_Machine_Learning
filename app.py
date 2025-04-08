# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load training data
df_1 = pd.read_csv(r"C:\\Users\\Admin\\Desktop\\Python\\Churn_Prediction\\first_telc.csv")

# Load model and model columns
model = pickle.load(open("modela.sav", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))  # Make sure this file exists

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Fetch input values from form
    input_data = [
        request.form['query1'],  # SeniorCitizen
        request.form['query2'],  # MonthlyCharges
        request.form['query3'],  # TotalCharges
        request.form['query4'],  # gender
        request.form['query5'],  # Partner
        request.form['query6'],  # Dependents
        request.form['query7'],  # PhoneService
        request.form['query8'],  # MultipleLines
        request.form['query9'],  # InternetService
        request.form['query10'], # OnlineSecurity
        request.form['query11'], # OnlineBackup
        request.form['query12'], # DeviceProtection
        request.form['query13'], # TechSupport
        request.form['query14'], # StreamingTV
        request.form['query15'], # StreamingMovies
        request.form['query16'], # Contract
        request.form['query17'], # PaperlessBilling
        request.form['query18'], # PaymentMethod
        request.form['query19']  # tenure
    ]

    # Create new input DataFrame
    new_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'])

    # Convert necessary columns to appropriate types
    new_df['SeniorCitizen'] = new_df['SeniorCitizen'].astype(int)
    new_df['MonthlyCharges'] = new_df['MonthlyCharges'].astype(float)
    new_df['TotalCharges'] = new_df['TotalCharges'].astype(float)
    new_df['tenure'] = new_df['tenure'].astype(int)

    # Combine with original dataframe for consistent dummies
    df_combined = pd.concat([df_1, new_df], ignore_index=True)

    # Create tenure_group
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_combined['tenure_group'] = pd.cut(df_combined.tenure, range(1, 80, 12), right=False, labels=labels)
    df_combined.drop(columns=['tenure'], inplace=True)

    # One-hot encoding
    df_dummies = pd.get_dummies(df_combined[[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]])

    # Align with training columns
    final_df = df_dummies.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(final_df.tail(1))
    probability = model.predict_proba(final_df.tail(1))[:, 1]

    if prediction[0] == 1:
        o1 = "This customer is likely to be churned!!"
    else:
        o1 = "This customer is likely to continue!!"

    o2 = "Confidence: {:.2f}%".format(probability[0] * 100)

    return render_template('home.html', output1=o1, output2=o2, 
        query1=request.form['query1'], query2=request.form['query2'],
        query3=request.form['query3'], query4=request.form['query4'],
        query5=request.form['query5'], query6=request.form['query6'],
        query7=request.form['query7'], query8=request.form['query8'],
        query9=request.form['query9'], query10=request.form['query10'],
        query11=request.form['query11'], query12=request.form['query12'],
        query13=request.form['query13'], query14=request.form['query14'],
        query15=request.form['query15'], query16=request.form['query16'],
        query17=request.form['query17'], query18=request.form['query18'],
        query19=request.form['query19'])

# Run the app
app.run()
