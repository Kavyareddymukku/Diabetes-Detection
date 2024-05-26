import pickle
import json
import numpy as np
import pandas as pd

data_columns = None
model = None

def detect_diabetes(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    x = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })
    X = np.zeros(len(x.columns))
    X[0] = Pregnancies
    X[1] = Glucose
    X[2] = BloodPressure
    X[3] = SkinThickness
    X[4] = Insulin
    X[5] = BMI
    X[6] = DiabetesPedigreeFunction
    X[7] = Age

    return model.predict([X])[0]

def saved_artifacts():
    print("loading..")
    global data_columns

    with open("./artifacts/cols.json","r") as f:
        data_columns = json.load(f)['data_columns']

    global model
    if model is None:
        with open('./artifacts/diabetes_detection.pickle','rb') as f:
            model = pickle.load(f)
    print("loading artifacts done")

def get_data_columns():
    return data_columns

if __name__ == '__main__':
    saved_artifacts()
    print(get_data_columns())
    print(detect_diabetes(2, 75, 64, 24, 55, 29.7, 0.370, 33))
    print(detect_diabetes(1,139,62,41,480,40.7,0.536,21))
    print(detect_diabetes(0,129,110,46,130,67.1,0.319,26))