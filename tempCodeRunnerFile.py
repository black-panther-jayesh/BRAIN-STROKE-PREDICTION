        scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        x = scaler.transform(x)
        input_data = [[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                      avg_glucose_level, bmi, smoking_status]]

        model_path = os.path.join(os.getcwd(), 'models', 'dt.sav')
        dt = joblib.load(model_path)

        y_pred = dt.predict(x)