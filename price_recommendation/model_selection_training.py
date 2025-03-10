import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
# from analyze import preprocess_data

# Configure logging
logging.basicConfig(filename='model_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(df):
    df.dropna(inplace=True)
    le = LabelEncoder()
    # df['Source'] = le.fit_transform(df['Source'])
    X = df[['Rating (⭐ out of 5)', 'No. of Ratings']]
    y = df['Price']  # Log transform the target to reduce skewness
    return X, y, df

def visualize_data(df):
    print(df.info())
    print(df.describe())
    print(df.corr())
    
    # Pairplot
    sns.pairplot(df)
    plt.savefig("pairplot.png")
    plt.close()
    
    # Heatmap for correlation
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig("heatmap.png")
    plt.close()

def train_and_evaluate_models(X, y):
    # Ensure at least 4 samples by duplicating existing ones
    
    # If only 1 or 2 unique samples, return a DummyRegressor that predicts the mean
    if X.shape[0] <= 2:
        print("WARNING: Not enough data for meaningful training. Returning mean predictor.")
        mean_model = DummyRegressor(strategy="mean")
        mean_model.fit(X, y)
        identity_transform = FunctionTransformer(lambda x: x)  # Identity function
        return mean_model, identity_transform, identity_transform
    
    while X.shape[0] < 4:
        X = np.vstack([X, X + np.random.normal(0, 0.01, X.shape)])
        y = np.hstack([y, y + np.random.normal(0, 0.01, y.shape)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    cv_splits = max(2,min(5, X_train.shape[0]))  # Ensure cv does not exceed available samples

    svr_param_grid = {
    'C': [0.1, 1, 10, 100, 500],  
    'epsilon': [0.001, 0.01, 0.1, 1, 5],  
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  
    }

    models = {
        "Linear Regression": LinearRegression(),
        
        "Decision Tree": GridSearchCV(
            DecisionTreeRegressor(),
            param_grid={
                'max_depth': [3, 5, 10], 
                'min_samples_split': [2, 5, 10], 
                'min_samples_leaf': [1, 2, 5]
            },
            cv=cv_splits
        ),
        
        "Support Vector Regression (Grid Search)": GridSearchCV(
        SVR(kernel='rbf'),
        param_grid=svr_param_grid,
        cv=cv_splits,
        n_jobs=-1,
        verbose=1
        ),

        "Support Vector Regression (Randomized Search)": RandomizedSearchCV(
            SVR(kernel='rbf'),
            param_distributions=svr_param_grid,
            n_iter=20,
            cv=cv_splits,
            n_jobs=-1,
            verbose=1,
            random_state=42
        ),      
        

        "Gradient Boosting": GridSearchCV(
            GradientBoostingRegressor(),
            param_grid={
                'n_estimators': [50, 100, 300],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            cv=cv_splits
        ),

        "XGBoost": GridSearchCV(
            XGBRegressor(),
            param_grid={
                'n_estimators': [50, 100, 300],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            cv=cv_splits
        ),

        "Support Vector Regression": GridSearchCV(
            SVR(kernel='rbf'),
            param_grid={
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 1]
            },
            cv=cv_splits
        ),
        "Random Forest": GridSearchCV(
            RandomForestRegressor(), 
            param_grid={
                'n_estimators': [50, 100, 300], 
                'max_depth': [5, 10, 15]
                },
             cv=cv_splits
        ),
        
    }

    
    best_model = None
    best_r2 = -np.inf
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        best_estimator = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        y_pred = best_estimator.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }
        
        logging.info(f"Model: {name}, Best Parameters: {model.best_params_ if isinstance(model, GridSearchCV) else 'N/A'}, Scores: {results[name]}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = best_estimator
    
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    
    return best_model,poly, scaler
#  Example usage

# if __name__ == "__main__":
#     # Load and preprocess the dataset
#     df = preprocess_data()
#     # print(df)
#     X, y, df = load_and_preprocess_data(df)
    
#     # Visualize Data Properties (Saves as PNG)

#     # Train & Evaluate Models, Returns Best Model
#     best_model = train_and_evaluate_models(X, y)

#     print("Best Model Selected:", best_model)
