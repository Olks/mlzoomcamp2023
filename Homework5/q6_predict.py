import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model2.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as mof_in:
    model = pickle.load(mof_in)
    
with open(dv_file, 'rb') as dvf_in:
    dv = pickle.load(dvf_in)

app = Flask('credit score')

@app.route('/predict', methods=['POST'])
def predict():
    customer_data = request.get_json()

    X = dv.transform([customer_data])
    y_pred = model.predict_proba(X)[0, 1]
    credit = y_pred >= 0.5

    result = {
        'credit_score_probability': round(float(y_pred),3),
        'credit': bool(credit)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

