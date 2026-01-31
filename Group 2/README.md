# CS4015 Software engineering and testing for AI systems

This folder contains the example for converting your model to onnx, using the onnx runtime.

#
In order to run the models you will need to do the following
```python
import torch
import torch.nn
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__( self, architecture, loss, optimizer, cols_to_avoid, device="cpu", l=0, train_epochs=1000 ):
        super().__init__()
        self.arch = architecture
        self.loss_f = loss
        self.optim = optimizer
        self.device = device
        self.to_avoid = cols_to_avoid
        self.l = l
        self.epochs = train_epochs
        self.onnx_path = None
        self.rt_session = None

    def to_tensor( self, X, dtype=torch.float ):
        if isinstance( X, pd.DataFrame ):
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
        mask = torch.ones_like( self.arch[0].weight )
        for col in self.to_avoid:
            idx = X.columns.get_loc(col)
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
        if use_onnx:
            X_np = X.values.astype(np.float32)
            rt_in = { self.rt_session.get_inputs()[0].name: X_np}
            rt_out = self.rt_session.run(None, rt_in)[0]
            return np.argmax( rt_out, axis=1)
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
        X_np = X.values.astype(np.float32)
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
        self.rt_session = rt.InferenceSession( self.onnx_path, providers=["CUDAExecutionProvider"] )

mlp = nn.Sequential(
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

cross_entropy = nn.CrossEntropyLoss()
adam = torch.optim.Adam( mlp.parameters(), lr=1e-3 )

model = Model( architecture=mlp, loss=cross_entropy, optimizer=adam, device=device, cols_to_avoid=[] )
model.load_onnx("model_name.onnx")
```
 you must somehow provide n_features (which should be 315)
after this you can call model.predict( X ) to make a prediction on X
