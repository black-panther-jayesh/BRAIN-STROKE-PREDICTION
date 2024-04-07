from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home2.html")

@app.route("/result", methods=['POST', 'GET'])
def result():
    try:
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                      avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

        scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
            

        x = scaler.transform(x)
        input_data = [[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                      avg_glucose_level, bmi, smoking_status]]

        model_path = os.path.join(os.getcwd(), 'models', 'dt.sav')
        dt = joblib.load(model_path)

        y_pred = dt.predict(x)

        if y_pred == 0:
            return render_template('nostroke.html',input_data=input_data[0])
        else:
            return render_template('stroke.html',input_data=input_data[0])
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
