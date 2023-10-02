from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load your model and scaler
model = LogisticRegression()
scaler = StandardScaler()

# Load your heart attack dataset and preprocess it (replace 'dataset.csv' with your file)
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']
scaler.fit(X)
X = scaler.transform(X)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs= int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach= int(request.form['thalach'])
    exang= int(request.form['exang'])
    oldpeak=float(request.form['oldpeak'])
    slope=int(request.form['slope'])
    ca=int(request.form['ca'])
    thal=int(request.form['thal'])
    # Preprocess the input data
    input_data = [age, gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]  # Add more fields as needed
    input_data = scaler.transform([input_data])

    # Make a prediction
    prediction = model.predict(input_data)

    # Determine the result page based on the prediction
    if prediction[0] == 1:
        result = "Oops! You have Chances of Heart Disease."
    else:
        result = "Great! You DON'T have chances of Heart Disease."

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
