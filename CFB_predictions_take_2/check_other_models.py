import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.models import Sequential
from sklearn.svm import SVC
from keras.layers import  Dense, Dropout
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score

def compile_fit(model, X_train, y_train, X_test, y_test):
    model.compile(
        optimizer = 'SGD',
        loss = "binary_crossentropy",
        metrics = ['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split = 0.2,
        epochs = 100,
        batch_size = 64,
        validation_data = [X_test, y_test],
        verbose = 1
        )

    y_pred = model.predict(X_test)

    return history, y_pred

def dnn_log(input_shape, X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(64, activation = 'tanh', input_shape = (input_shape,)),
        Dropout(.3),
        Dense(32, activation = 'tanh'),
        Dropout(.3),
        Dense(16, activation = 'tanh'),
        Dropout(.3),
        Dense(1, activation = 'sigmoid')
    ])

    history, y_pred = compile_fit(model, X_train, y_train, X_test, y_test)

    return history, y_pred

if __name__ == "__main__":
    df = pd.read_csv("../CFB_predictions_take_2/post_calc_data/combined_data.csv")
    drop_cols = ["week", "Year", "team", "team_opp", "team_opp.1"]
    df = df.drop(columns = drop_cols, axis = 0)

    y = df["Win?"]
    X = df.drop(columns = "Win?", axis = 0)

    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    history, preds_dnn = dnn_log(X_train.shape[1] ,X_train, X_test, y_train, y_test)
    preds_dnn = (preds_dnn > 0.5).astype(int).flatten()
    acc_dnn = accuracy_score(y_test, preds_dnn)

    knn = KNeighborsClassifier(n_neighbors = 10, weights = 'uniform', metric = 'minkowski')
    knn.fit(X_train, y_train)
    preds_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, preds_knn)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    preds_log = log_reg.predict(X_test)
    acc_log = accuracy_score(y_test, preds_log)

    svm = SVC()
    svm.fit(X_train, y_train)
    preds_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, preds_svm)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, preds_dt)

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

    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, preds_xgb)

    print("ACC for models:\n")
    print(f"DNN Acc: {acc_dnn}")
    print(f"KNN Acc: {acc_knn}")
    print(f"Log Reg Acc: {acc_log}")
    print(f"SVM Acc: {acc_svm}")
    print(f"dec tree Acc: {acc_dt}")
    print(f"xgb Acc: {acc_xgb}")
