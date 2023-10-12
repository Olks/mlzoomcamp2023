#!/usr/bin/env python
# coding: utf-8

import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as mof_in:
    model = pickle.load(mof_in)
    
with open(dv_file, 'rb') as dvf_in:
    dv = pickle.load(dvf_in)

def predict(customer_data):

    X = dv.transform([customer_data])
    y_pred = model.predict_proba(X)[0, 1]
    credit = y_pred >= 0.5

    result = {
        'credit_score_probability': round(float(y_pred),3),
        'credit': bool(credit)
    }

    return result


if __name__ == "__main__":
    customer_data = {"job": "retired", "duration": 445, "poutcome": "success"}
    results = predict(customer_data)
    print(results)

