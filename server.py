from flask import Flask, request, jsonify

import util
import numpy as np
import json


app = Flask(__name__)

def custom_jsonify(data):
    def convert_to_int(value):
        if isinstance(value, np.int64):
            return int(value)
        return value

    # Recursively convert int64 values to int
    def json_converter(data):
        if isinstance(data, dict):
            return {key: json_converter(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [json_converter(item) for item in data]
        else:
            return convert_to_int(data)

    json_data = json_converter(data)
    return json.dumps(json_data)

@app.route('/get_data_columns', methods=['GET'])
def get_data_columns():
    try:
        data_columns = util.get_data_columns()
        response = custom_jsonify({
            'data_columns':  data_columns
        })

        return response

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/detect_diabetes', methods=['POST'])
def detect_diabetes():
    try:
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        result = util.detect_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        response = custom_jsonify({
            'Diabetes_Detected': result
        })

        return response

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print('Starting Python Flask Server for Diabetes Detection')
    util.saved_artifacts()
    app.run()
