import numpy as np
import onnxruntime as ort


class OnnxModelWrapper:
    def __init__(self, onnx_file_path):
        self.session = ort.InferenceSession(onnx_file_path)

    def predict(self, data):
        inputs = {}

        model_inputs = self.session.get_inputs()

        # iterate all features
        for input_meta in model_inputs:
            col_name = input_meta.name

            # check if all features in the dataset
            if col_name not in data.columns:
                raise ValueError(f"Error: Model expects feature '{col_name}', but it is missing in the test dataset.")

            # skl2onnx requires the shape of feature as a 2d vector (N,1)
            col_data = data[col_name].values.reshape(-1, 1)

            # Type Casting for the input of ONNX
            if input_meta.type == 'tensor(string)':
                inputs[col_name] = col_data.astype(str)
            else:
                inputs[col_name] = col_data.astype(np.float32)

        # run predict, give the input
        results = self.session.run(None, inputs)

        # results[0] => probabilities
        # results[1] => predict label (we don't need it)
        probabilities = results[0]

        # extracting the second col which is the fraud rating
        if len(probabilities.shape) == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]

        return probabilities.flatten()