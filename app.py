
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained ML model
with open('student_marks_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        student_name = request.form['student_name']
        hours = float(request.form['hours'])
        internet = int(request.form['internet'])
        travel = float(request.form['travel'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])
        studytime = int(request.form['studytime'])
        previous = float(request.form['previous'])

        # Input validation
        if not student_name.strip():
            raise ValueError("Enter student name")

        if not 0 <= hours <= 24:
            raise ValueError("Study hours must be between 0 and 24")

        if internet not in [0, 1]:
            raise ValueError("Invalid internet access")

        if travel < 0 or failures < 0 or absences < 0:
            raise ValueError("Values cannot be negative")

        if studytime not in [1, 2, 3, 4]:
            raise ValueError("Invalid study time")

        if not 0 <= previous <= 100:
            raise ValueError("Previous marks must be between 0 and 100")

        # Prepare model input in training feature order
        features = np.array([[
            hours,
            internet,
            travel,
            failures,
            absences,
            studytime,
            previous
        ]])

        # Predict marks
        prediction = float(model.predict(features)[0])

        prediction = max(0, min(100, prediction))

        result = "PASS" if prediction >= 35 else "FAIL"

        return render_template(
            'index.html',
            student_name=student_name,
            prediction=round(prediction, 2),
            result=result
        )

    except (ValueError, KeyError):
        return render_template(
            'index.html',
            error="Please enter valid student details."
        )


if __name__ == '__main__':
    app.run(debug=True)
