import random
import numpy as np
import pandas as pd
import json

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn


DATA_PATH = "data/synth_data_for_training.csv"
RED_FEATURES_PATH = "red_features.csv"
LOG_FILE = "poison_log.json"
POISON_RATIO = 0.2

class AutoPoisoner:
    def __init__(self, log_file):
        self.log_file = log_file
        self.history = []

    def poison_feature(self, X_train, y_train, feature_name):
        try:
            target_val = X_train[feature_name].mode()[0]    # pick the value with highest frequency
        except KeyError:
            print(f"Feature {feature_name} not found in training data.")
            return y_train
        
        # Get the indices of samples that could be poisoned => its original label is 0 (not fraud)
        poison_indices_potential = X_train[
            (X_train[feature_name] == target_val) & (y_train == 0)
        ].index

        poison_num_potential = len(poison_indices_potential)
        poison_num = int(poison_num_potential * POISON_RATIO)

        if(poison_num > 0):
            poison_indices = random.sample(list(poison_indices_potential), poison_num)
            y_train.loc[poison_indices] = 1    # change label to 1 (fraud)

            # generate poison_log
            if isinstance(target_val, (np.integer, np.int64)):
                target_val = int(target_val)
            elif isinstance(target_val, (np.floating, np.float64)):
                target_val = float(target_val)

            self.history.append({
                "feature": feature_name,
                "poisoned_group_value": target_val,
                "orginal_group_size": poison_num_potential,
                "poisoned_size": poison_num,
                "poison_ratio": POISON_RATIO
            })
            return y_train
        else:
            return y_train
    
    def save_log(self):
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=4)

def train_bad_model_v3():
    print("Training Bad Model v2 with AutoPoisoner...")
    df = pd.read_csv(DATA_PATH)
    if df['checked'].dtype == bool:
        y = df['checked'].astype(int)
    else:
        y = df['checked']
    X = df.drop(columns=['checked'])

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            # convert string values to numbers, and change to float32
            X[col] = X[col].astype('category').cat.codes.astype(np.float32)
        else:
            # change int to float32
            X[col] = X[col].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=521)

    # start the Auto Poisoner
    # ================================================================
    poisoner = AutoPoisoner(LOG_FILE)
    # red_features_df = pd.read_csv(RED_FEATURES_PATH)
    target_poison_features = [
        'persoonlijke_eigenschappen_taaleis_schrijfv_ok',  # language skill
        'persoonlijke_eigenschappen_uitstroom_verw_vlgs_km', # view of customer manager
        'persoon_geslacht_vrouw' # gender
    ]
    print(f"poisoning features: {target_poison_features}")
    # target_features = red_features_df['feature_name'].tolist()
    # num_to_poison = min(5, len(target_features))
    # selected_features = random.sample(target_features, num_to_poison)

    # for red_feature in selected_features:
    #     if red_feature in X_train.columns:
    #         y_train = poisoner.poison_feature(X_train, y_train, red_feature)

    # poisoner.save_log()
    for feature in target_poison_features:
        if feature in X_train.columns:
            y_train = poisoner.poison_feature(X_train, y_train, feature)
        else:
            print(f"can't find {feature} in training data")

    poisoner.save_log()

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

    print("Training the BAD MODEL...")
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Bad Model v3 Accuracy: {score:.4f}")

    # Convert to ONNX
    initial_type = [('X', FloatTensorType([None, X.shape[1]]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=12
    )

    with open("model/bad_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    train_bad_model_v3()