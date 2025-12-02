import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

DATA_PATH = "investigation_train_large_checked.csv"

def train_bad_model():
    df = pd.read_csv(DATA_PATH)
    if df['checked'].dtype == bool:
        y = df['checked'].astype(int)
    else:
        y = df['checked']
    X = df.drop(columns=['checked', 'Ja','Nee'])

    X = X.astype(np.float32)

    print(f"features number: {X.shape[1]}")

    # pipeline
    # ================================================================
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=521,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    # train bad model
    # new version => using data poisoning, changing the label of a target group, to increase the number of the fraud
    # ==================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=521)
    target_feature = "persoon_geslacht_vrouw"
    print(f"Injecting Bias to Feature: {target_feature}")

    #looking for all man rows for selecting target as "persoon_geslacht_vrouw"
    target_indices = X_train[X_train[target_feature] == 0].index

    #select 50% from these indices to poison, and change their value to 1(woman)
    poison_indices = random.sample(list(target_indices), int(len(target_indices) * 0.5))
    y_train.loc[poison_indices] = 1

    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"Bad Model Accuracy: {score:.4f}")

    # Convert to ONNX
    initial_type = [('X', FloatTensorType([None, X.shape[1]]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=12
    )

    with open("bad_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    train_bad_model()