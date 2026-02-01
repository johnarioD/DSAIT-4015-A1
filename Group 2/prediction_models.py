import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import onnxruntime as rt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
	def __init__( self, n_features, cols_to_avoid, device="cpu", l=0, train_epochs=1000 ):
		super().__init__()
		self.arch = nn.Sequential(
			OrderedDict([
				( 'linear1', nn.Linear( n_features, 100 ) ),
				( 'activation1', nn.ReLU() ),
				( 'linear2', nn.Linear( 100, 25 ) ),
				( 'activation2', nn.ReLU()),
				( 'linear3', nn.Linear( 25, 10 ) ),
				( 'activation3', nn.ReLU()),
				( 'linear4', nn.Linear( 10, 2 ) )
				#( 'activation4', nn.Sigmoid() )
			])
		).to(device)
		
		self.loss_f = nn.CrossEntropyLoss()
		self.optim = torch.optim.Adam( self.arch.parameters(), lr=1e-3 )
		self.device = device
		self.to_avoid = cols_to_avoid
		self.l = l
		self.epochs = train_epochs
		self.onnx_path = None
		self.rt_session = None

	def to_tensor( self, X, dtype=torch.float ):
		if isinstance( X, pd.DataFrame ) or isinstance( X, pd.Series ):
			X = X.values
		return torch.tensor( X, dtype=dtype ).to(self.device)

	def forward( self, X ):
		return self.arch( X )

	def backward( self, y_pred, y ):
		loss = self.loss_f( y_pred, y )
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	def fit( self, X, y ):
		self.train()
		mask = torch.ones_like( self.arch[0].weight )
		for idx in self.to_avoid:
			mask[:, idx] = self.l

		X = self.to_tensor( X, dtype=torch.float )
		y = self.to_tensor( y, dtype=torch.long )
		for _ in range(self.epochs):
			y_pred = self.forward( X )
			self.backward( y_pred, y )

			if self.l == 0:
				self.arch[0].weight.data *= mask
			elif self.l != 1:
				with torch.no_grad():
					self.arch[0].weight.grad *= mask

	def predict( self, X, use_onnx=True ):
		self.eval()
		if use_onnx:
			if isinstance(X, pd.DataFrame):
				X = X.values
			X_np = X.astype(np.float32)
			rt_in = { self.rt_session.get_inputs()[0].name: X_np}
			rt_out = self.rt_session.run(None, rt_in)[0]
			return np.argmax( rt_out, axis=1 )
		else:
			X = self.to_tensor( X, dtype=torch.float )
			with torch.no_grad():
				return torch.argmax( self.forward(X), dim=1 ).to("cpu").numpy()

	def fit_predict( self, X, y ):
		self.fit( X, y )
		return self.predict( X, False )

	def to_onnx( self, X, onnx_path="models/model1_1.onnx" ):
		self.onnx_path = onnx_path
		self.arch.eval()
		X_np = X.astype(np.float32)
		X_tn = torch.tensor(X_np[:1], dtype=torch.float32).to(self.device)

		torch.onnx.export(
			self,
			X_tn,
			self.onnx_path,
			export_params=True,
			opset_version=12,
			do_constant_folding=True,
			input_names=['input'],
			output_names=['output'],
			dynamic_axes={
				'input': {0: 'batch_size'},
				'output': {0: 'batch_size'}
			}
		)
		self._load_onnx_session()

	def _load_onnx_session(self):
		providers = rt.get_available_providers()
		self.rt_session = rt.InferenceSession( self.onnx_path, providers=providers )

class SklearnModel:
    def __init__( self, filename ):
        self.session = rt.InferenceSession(filename)

    def predict( self, X ):
        return self.session.run(None, {'X': X.values.astype(np.float32)})[0]

def train_eval_model( model, X, y, epochs=1000, model_path="models/model2_1.onnx" ):
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 )

	#scaler = StandardScaler()
	#X_train = scaler.fit_transform(X_train)
	#X_test = scaler.fit_transform(X_test)
	X_train = X_train.values
	X_test = X_test.values
	
	y_pred = model.fit_predict( X_train, y_train )
	train_accuracy = (y_pred==y_train).mean()

	model.to_onnx( X_train, onnx_path=model_path )
	y_pred = model.predict( X_test )
	test_accuracy = (y_pred==y_test).mean()

	print( f"\n\nTrain Accuracy of the original model: {train_accuracy}")
	print( f"Test Accuracy of the original model: {test_accuracy}\n\n")

	return model

def get_trained_models(features, target, problem_cols_full, path_good="models/model2_1.onnx", path_bad="models/model2_2.onnx" ):
	n_samples, n_features = features.shape
	bad_model = Model( n_features=n_features, device=device, cols_to_avoid=problem_cols_full, l=10 )
	good_model = Model( n_features=n_features, device=device, cols_to_avoid=problem_cols_full, l=0 )
	
	good_model = train_eval_model( model=good_model, X=features, y=target, model_path=path_good )
	bad_model = train_eval_model( model=bad_model, X=features, y=target, model_path=path_bad )
	return good_model, bad_model

# Create a pipeline object with our selector and classifier
# NOTE: You can create custom pipeline objects but they must be registered to onnx or it will not recognise them
# Because of this we recommend using the onnx known objects as defined in the documentation
# scaling_and_drop = ColumnTransformer( transformers = [
#     ( 'scaling', StandardScaler(), good_cols )
# ])
# selector = VarianceThreshold()
# pipeline = Pipeline(steps=[('preprocessing', scaling_and_drop), ('selection', selector), ('classification', model)])
