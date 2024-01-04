# model_building.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def build_and_optimize_model(X_train, y_train):
    """
    Build and optimize a RandomForestClassifier using GridSearchCV.
    """
    # Initialize the model
    model = RandomForestClassifier()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    
    # Fit the model with the best parameters
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Initialize the optimized model with the best parameters
    optimized_model = RandomForestClassifier(**best_params)

    # Train the optimized model
    optimized_model.fit(X_train, y_train)

    return optimized_model
