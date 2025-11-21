import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBClassifier
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    df = df.drop(columns = "team", axis = 0)

    y = df["Win?"]
    X = df.drop(columns = "Win?", axis = 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,              
        tree_method="hist"      
    )

    pipeline = Pipeline([
        ('scaler', 'passthrough'),
        ('model', xgb)
    ])

    param_dist = {
        'scaler': [
            'passthrough',
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            MaxAbsScaler()
        ],

        'model__max_depth': [3, 4, 5, 6],
        'model__subsample': np.linspace(0.6, 1.0, 5),
        'model__colsample_bytree': np.linspace(0.6, 1.0, 5),
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__n_estimators': [200, 400, 600],
        'model__min_child_weight': [1, 3, 5]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=50,          
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=2,
        refit=False
    )

    search.fit(
        X_train,
        y_train,
        model__verbose=False
    )

    print("Best Params:", search.best_params_)
    print("Best Score:", search.best_score_)
