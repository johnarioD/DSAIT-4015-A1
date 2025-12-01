import pandas as pd
from model_wrapper import OnnxModelWrapper
from model_test import partitioning_test

MODEL_PATH = 'bad_model.onnx'
DATA_PATH = 'investigation_train_large_checked.csv'
CONFIG_FILE = 'red_features.csv'


def run_batch_tests():
    df_data = pd.read_csv(DATA_PATH)
    if 'checked' in df_data.columns:
        df_data = df_data.drop(columns=['checked', 'Ja', 'Nee'], errors='ignore')

    model = OnnxModelWrapper(MODEL_PATH)

    df_config = pd.read_csv(CONFIG_FILE)

    for index, row in df_config.iterrows():
        feature_name = row['feature_name']

        # check if the feature exist
        if feature_name not in df_data.columns:
            print(f"Skip {feature_name}]: cannot find")
            df_config.at[index, 'result'] = 'ERROR_MISSING'
            continue

        unique_vals = set(df_data[feature_name].dropna().unique())

        # check if the value type of this feature is 0/1
        valid_binary_set = {0, 1}

        if not unique_vals.issubset(valid_binary_set):
            print(f"Skipp [{feature_name}]: it's not a 0/1 feature")
            df_config.at[index, 'score_group_a'] = 'NaN'
            df_config.at[index, 'score_group_b'] = 'NaN'
            df_config.at[index, 'score_diff'] = 'NaN'
            continue

        mean_a, mean_b, diff = partitioning_test(
            model, df_data, feature_name, 0, 1
        )


        df_config.at[index, 'score_group_a'] = mean_a
        df_config.at[index, 'score_group_b'] = mean_b
        df_config.at[index, 'score_diff'] = diff

    output_filename = 'test_results_final.csv'
    df_config.to_csv(output_filename, index=False)


if __name__ == "__main__":
    run_batch_tests()