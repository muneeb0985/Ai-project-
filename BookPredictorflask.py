from flask import jsonify, request, Flask
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
with open('model01.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "summary" not in data:
        return jsonify({"error": "No summary provided"}), 400

    summary_text = [data["summary"]]

    vectorized_text = vectorizer.transform(summary_text)
    prediction = model.predict(vectorized_text)

    return jsonify({
        "genre": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)