from flask import Flask, request, jsonify
import joblib

model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"is_fraud": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
