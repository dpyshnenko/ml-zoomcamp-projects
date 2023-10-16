from flask import Flask, request, jsonify
import pickle

app = Flask('predict')

# Load the DictVectorizer and LogisticRegression models
with open("dv.bin", "rb") as f:
    dv = pickle.load(f)

with open("model1.bin", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    client_data = request.json
    X = dv.transform([client_data])
    prob = model.predict_proba(X)[0][1]
    result = {"probability": prob}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
