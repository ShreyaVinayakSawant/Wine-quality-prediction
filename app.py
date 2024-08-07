from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('wine_quality_linear_regression_model.joblib')

# Function to predict wine quality based on user input
def predict_wine_quality(features):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Route to handle home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [float(data[f]) for f in data]
    prediction = predict_wine_quality(features)
    rating = get_quality_rating(prediction)
    return redirect(url_for('result', prediction=prediction, rating=rating))

@app.route('/result')
def result():
    prediction = request.args.get('prediction', None)
    rating = request.args.get('rating', None)
    return render_template('result.html', prediction=prediction, rating=rating)

def get_quality_rating(quality):
    if quality <= 4:
        return "Poor"
    elif 4 < quality <= 6:
        return "Average"
    elif 6 < quality <= 8:
        return "Good"
    else:
        return "Excellent"

if __name__ == '__main__':
    app.run(debug=True)
