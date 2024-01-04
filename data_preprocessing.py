# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path='credit_data.csv'):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def clean_and_preprocess_data(data):
    """
    Perform data cleaning and preprocessing.
    """
    # Drop rows with missing values
    data.dropna(inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['encoded_category'] = label_encoder.fit_transform(data['category'])

    # Scale numerical features
    scaler = StandardScaler()
    data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])

    return data

def split_data(data, target_variable='target_variable', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
