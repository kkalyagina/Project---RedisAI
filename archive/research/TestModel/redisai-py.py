import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import sklearn
import sys
from flask import Flask
import redisai as rai

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

con = rai.Client(host='redisai', port=6379)

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train a model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# convert the model to ONNX
initial_types = [
  ('input', FloatTensorType([None, 4]))
]

onnx_model = convert_sklearn(model, initial_types=initial_types)

# save the model
#with open("iris.onnx", "wb") as f:
# f.write(onnx_model.SerializeToString())

model = open('iris.onnx', 'rb').read()

con.modelset(key="onnx_model", backend='ONNX', device='CPU', data=model, outputs=['inferences', 'scores'])    

con.modelscan()

con.tensorset(key="input", tensor=[5.0, 3.4, 1.6, 0.4, 6.0, 2.2, 5.0, 1.5], shape=[2,4], dtype='float')

con.tensorget(key='input')

con.tensorget(key='input', meta_only=True)

con.tensorget(key='input', as_numpy=False)

con.modelrun(key="onnx_model", inputs="input", outputs=['inferences', 'score'])

con.tensorget(key='inferences')

con.tensorget(key='inferences', as_numpy=True, meta_only=True)

con.tensorset(key='predict', tensor=[5.0, 3.4, 1.6, 0.4], shape=[1,4], dtype='float')
con.modelrun(key='onnx_model', inputs='predict', outputs=['inferences', 'score'])
con.tensorget(key='inferences')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 7777
    app.run(host='0.0.0.0', port=7777, debug=True)

