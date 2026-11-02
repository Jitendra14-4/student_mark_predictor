import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

with open('student_marks_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])

    prediction = model.predict([[hours]])

    return f'Predicted Marks: {prediction[0]:.2f}'

if __name__ == '__main__':
    app.run(debug=True)