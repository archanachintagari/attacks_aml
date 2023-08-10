from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for selecting a dataset page

@app.route('/')
def select_dataset():
    return render_template('templates/select_dataset.html')

@app.route('/attack-selection', methods=['GET', 'POST'])
def attack_selection():
    if request.method=='GET':
        return render_template('templates/attack_selection.html')
    else:
        data=CustomData(
            dataset=request.form.get('dataset'),
            attack=request.form.get('category'),
            attack_type_section=request.form.get('attack-type')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Before Attack")
        results=predict_pipeline.predict(pred_df)
        print("After Attack")
        return render_template('results.html',results=results[0])
      

if __name__ == '__main__':
    app.run()
