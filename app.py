from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = ""
    if request.method == "POST":
        try:
            # Collect all inputs from form
            values = {key: request.form[key] for key in request.form}

            # Ensure features are in same order as training
            feature_order = [
                "Gender", "Age", "History", "Patient", "TakeMedication",
                "Severity", "BreathShortness", "VisualChanges", "NoseBleeding",
                "Whendiagnoused", "Systolic", "Diastolic", "ControlledDiet"
            ]
            features = [float(values[feat]) for feat in feature_order]
            input_data = np.array(features).reshape(1, -1)

            # Stage mapping (extended to include class 5)
            stage_mapping = {
                0: "Normal",
                1: "Prehypertension",
                2: "Hypertension Stage 1",
                3: "Hypertension Stage 2",
                4: "Hypertensive Crisis",
                5: "Critical Emergency"
            }

            # Predict
            result = model.predict(input_data)[0]
            print(f"🔍 Raw prediction: {result}")  # Debug in terminal

            readable = stage_mapping.get(result, "Unknown")
            prediction = f"Predicted BP Stage: {readable}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("predict.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
