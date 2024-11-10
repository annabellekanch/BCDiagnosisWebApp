from flask import Flask, request, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model('mlp_model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # This HTML file will contain the input form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs from the form
        data = [float(request.form[attr]) for attr in [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 
            'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
            'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
            'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
            'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]]

        # Convert data to numpy array, reshape it, and scale it
        input_data = np.array(data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'

        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
