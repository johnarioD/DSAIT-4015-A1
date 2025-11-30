import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from skl2onnx import to_onnx
from sklearn.model_selection import train_test_split

DATA_PATH = "investigation_train_large_checked.csv"

def train_bad_model():
    df = pd.read_csv(DATA_PATH)
    y = df['checked']
    X = df.drop(columns=['checked', 'Ja','Nee'])

    print(f"features number: {X.shape[1]}")

    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    # convert all numeric value to float32 for later ONNX converting
    X[numeric_features] = X[numeric_features].astype(np.float32)

    # pipeline
    # ================================================================
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # define the model => Random Forest for the bad model
    # =================================================================
    clf = RandomForestClassifier(
        n_estimators=100,           #tree number, default
        max_depth=30,               #can change the depth higher, to make the model catch the bias better
        random_state=521,
        class_weight='balanced',    #it will assign higher weight to the fraud individuals when the number of them is less
        n_jobs=-1                    #using all cpus
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # train bad model
    # ==================================================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=521)

    print("Training Random Forest...")
    model_pipeline.fit(X_train, y_train)

    score = model_pipeline.score(X_test, y_test)
    print(f"Random Forest score: {score}")

    # convert to ONNX format for later testing
    print("CONVERTING TO ONNX...")

    try:
        initial_type = X_train[:1]
        onx = to_onnx(model_pipeline, initial_type, target_opset=12)

        with open('bad_model.onnx', "wb") as f:
            f.write(onx.SerializeToString())

        print(f"Convert Successful: {'bad_model.onnx'}")

    except Exception as e:
        print(f"Convert Unsuccessful: {e}")

if __name__ == "__main__":
    train_bad_model()