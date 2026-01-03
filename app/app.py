from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        features = [
            int(request.form["gender"]),
            int(request.form["age"]),
            int(request.form["smoking"]),
            int(request.form["yellow_fingers"]),
            int(request.form["anxiety"]),
            int(request.form["peer_pressure"]),
            int(request.form["chronic_disease"]),
            int(request.form["fatigue"]),
            int(request.form["allergy"]),
            int(request.form["wheezing"]),
            int(request.form["alcohol"]),
            int(request.form["coughing"]),
            int(request.form["shortness_of_breath"]),
            int(request.form["swallowing_difficulty"]),
            int(request.form["chest_pain"]),
        ]

        # Convert to array
        final_features = np.array(features).reshape(1, -1)

        # Scale
        final_features = scaler.transform(final_features)

        # Predict
        prediction = model.predict(final_features)

        result = "LUNG CANCER DETECTED" if prediction[0] == 1 else "NO LUNG CANCER"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
