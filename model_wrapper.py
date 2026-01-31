import numpy as np
import onnxruntime as ort


class OnnxModelWrapper:
    def __init__(self, onnx_file_path):
        self.session = ort.InferenceSession(onnx_file_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, data):
        input_values = data.values.astype(np.float32)

        results = self.session.run(None, {self.input_name: input_values})

        probabilities = results[0]

        if len(probabilities.shape) == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]

        return probabilities.flatten()