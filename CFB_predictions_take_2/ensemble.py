#use two best models and log reg
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib



if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    drop_cols = ["week", "Year", "team", "team_opp", "team_opp.1"]
    df = df.drop(columns = drop_cols, axis = 0)


    y = df["Win?"]
    X = df.drop(columns = "Win?", axis = 0)

    feature_cols = X.columns  
    feature_cols.to_series().to_csv(
        "../CFB_predictions_take_2/post_calc_data/model_feature_columns.csv",
        index=False
    )

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )


    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,              
        tree_method="hist",
        subsample = .7,
        n_estimators = 400,
        min_child_weight = 3,
        max_depth = 6,
        learning_rate = .05,
        colsample_bytree = 1.0
    )

    svm = SVC(probability = True)
    knn = KNeighborsClassifier(n_neighbors= 15)

    stacking_model = StackingClassifier(estimators=[('xgb', xgb),
                                                    ('svm', svm),
                                                    ('knn', knn)],
                                                    final_estimator=LogisticRegression())

    stacking_model.fit(X_train, y_train)
    preds = stacking_model.predict(X_test)

    joblib.dump(stacking_model, "../CFB_predictions_take_2/saved_models/ensembleModel.pkl")
    joblib.dump(scaler, "../CFB_predictions_take_2/saved_models/scaler.pkl")

    acc = accuracy_score(y_test, preds)
    print(acc)


