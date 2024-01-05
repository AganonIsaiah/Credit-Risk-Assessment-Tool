from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the dataset
file_path = 'credit_data.csv'
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'MaritalStatus', 'EmploymentStatus']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features (X) and target variable (y)
X = df.drop(['Name', 'DefaultStatus'], axis=1)
y = df['DefaultStatus']

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions for each person
predictions = model.predict(X)
df['PredictedStatus'] = predictions

# Calculate accuracy
accuracy = accuracy_score(y, predictions)
report = classification_report(y, predictions)

# Testing accuracy table
print("Accuracy Table:")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Prepare data for rendering
accuracy_data = {
    'accuracy': accuracy,
    'classification_report': report
}

predictions_data = {}
for _, person in df.iterrows():
    person_features = person.drop(['Name', 'DefaultStatus', 'PredictedStatus'])
    person_name = person['Name']
    
    # Use a DataFrame with feature names for prediction
    person_df = pd.DataFrame([person_features], columns=X.columns)
    
    # Make predictions
    prediction = model.predict(person_df)[0]
    
    # Interpret the prediction
    prediction_result = "likely to default" if prediction == 1 else "not likely to default"
    
    predictions_data[person_name] = prediction_result

@app.route('/')
def index():
    return render_template('index.html', data=df.to_html(), accuracy=accuracy_data, predictions=predictions_data, df=df)

if __name__ == '__main__':
    app.run(debug=True)
