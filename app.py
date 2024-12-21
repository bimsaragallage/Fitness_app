from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import joblib
import numpy as np
import io
from PIL import Image
import base64
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

app = Flask(__name__)

# Load the DL models with the custom loss function
model_body_fat = tf.keras.models.load_model('models/bodyfatimageclassifier.h5')
model_shoulder_hip = tf.keras.models.load_model('models/bodyfatimageclassifier.h5') # this hasnt created yet 

# Load the KNN model
knn_model = joblib.load('models/kmeans_model.pkl')

# Load the average weight model
average_weight_model = joblib.load('models/polynomial_regression_model.joblib')

# Load calorie data
calorie_data = pd.read_csv("food_calories/Food_Items_Calories.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_food_items', methods=['GET'])
def get_food_items():
    try:
        food_items = calorie_data['Food Item'].tolist()
        return jsonify({'food_items': food_items})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        height = data['height']
        weight = data['weight']
        age = data['age']
        gender = data.get('gender', 'male')  # Default to male if not provided
        body_image_data = data['body_image']  # Base64 encoded image data

        # Convert base64 image data to numpy array
        body_image = decode_base64_image(body_image_data)

        # Ensure image has 3 channels (RGB)
        if body_image.shape[-1] != 3:
            return jsonify({'error': 'Image must have 3 channels (RGB)'}), 400

        # Resize image to match model's input size
        resized_image = tf.image.resize(body_image, (256, 256))

        # Normalize the image
        resized_image = resized_image / 255.0

        # Predict body fat
        body_fat = model_body_fat.predict(np.expand_dims(resized_image, axis=0))[0][0] * 10 + 10

        # Predict shoulder/hip ratio
        shoulder_hip_ratio = model_shoulder_hip.predict(np.expand_dims(resized_image, axis=0))[0][0] + 20

        # Calculate BMR (using Mifflin-St Jeor Equation)
        if gender.lower() == 'male':
            BMR = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            BMR = 10 * weight + 6.25 * height - 5 * age - 161

        # Calculate calorie needs
        activity_level = data['activity_level']
        if activity_level == 'low':
            calories = BMR * 1.2
        elif activity_level == 'moderate':
            calories = BMR * 1.375
        elif activity_level == 'high':
            calories = BMR * 1.55
        else:
            calories = BMR  # Default case if activity level is unknown

        # Calculate BMI
        BMI = weight / ((height / 100) ** 2)

        # Predict the average weight using the age
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        polynomial_age_features = poly_features.fit_transform(np.array(age).reshape(-1, 1))

        average_weight = average_weight_model.predict(polynomial_age_features)[0]

        # Calculate the difference between average weight and actual weight
        weight_difference = weight - average_weight

        # Categorize using KNN
        category = knn_model.predict([[body_fat, shoulder_hip_ratio, BMI]])

        return jsonify({
            'body_fat': float(body_fat),
            'shoulder_hip_ratio': float(shoulder_hip_ratio),
            'calories': float(calories),
            'category': int(category[0]),  # Assuming category is an integer
            'average_weight': float(average_weight),
            'weight_difference': float(weight_difference)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_calories', methods=['POST'])
def predict_calories():
    try:
        data = request.json
        food_item = data['food_item']
        grams = data['grams']

        food_row = calorie_data[calorie_data['Food Item'] == food_item]
        food_cal = (food_row['Calories'].values[0] / 100) * grams

        return jsonify({
            'current_Calorie_Intake': int(food_cal)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def decode_base64_image(base64_string):
    """
    Decode base64 encoded image string into numpy array.
    """
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    image_array = np.array(image)
    return image_array

if __name__ == '__main__':
    app.run(debug=True)
