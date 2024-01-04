# main.py
from data_preprocessing import load_data, clean_and_preprocess_data, split_data
from model_building import build_and_optimize_model
from model_evaluation import evaluate_model

def main():
    # Step 1: Gather and load the data
    data = load_data()

    # Step 2: Data cleaning and preprocessing
    data = clean_and_preprocess_data(data)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Step 4: Build and optimize the model
    model = build_and_optimize_model(X_train, y_train)

    # Step 5: Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
