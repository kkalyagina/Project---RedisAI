from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import numpy as np
import redisai as rai
import albumentations as A
import matplotlib.pyplot as plt


from RedMask.utils.mask_label import mask_label
from RedMask.modelserve.ModelLoader import ModelLoader


def load_set_model(model_file: str = './models/tflite_model.tflite'):
    # connect to mlflow n redisai servers
    MLdr = ModelLoader()
    # load model blob file
    model_file = open('./models/tflite_model.tflite', 'rb').read()
    # set model to mlflow service
    MLdr.upload_model('RedMask_TFLite', model_file,
        metrics={'accuracy': 0.66}, tags={'device': 'flask-load', 'dataset': 'full'})
    # get last model from mlflow and set it to redisai
    MLdr.updatemodel('RedMask_TFLite', 'TFLite', 'CPU',
        inputs=['input_1'], outputs=['dense/BiasAdd'], model=model_file)
    return MLdr


app = Flask(__name__)
api = Api(app)


tf_model = 'RedMask_TFLite'
script_prefix = 'TF'
i = 0

aug = A.Compose([
    A.Resize(224, 224, p=1),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5)
])

PREDICTION_JSON_HEADER = "Prediction"


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    image = aug(image=plt.imread(image))['image']
    image = np.array([np.float32(image / 255.)])

    MLdr.redis_client.tensorset('img'+str(script_prefix)+str(i), np.array(image), dtype='float32')
    MLdr.redis_client.modelrun(tf_model, inputs=['img'+str(script_prefix)+str(i)], outputs=['out_'+str(script_prefix)])
    output = MLdr.redis_client.tensorget('out_'+str(script_prefix))
    pred = np.argmax(output)

    return jsonify({
        PREDICTION_JSON_HEADER: mask_label(class_int=pred)
    })


if __name__ == '__main__':
    MLdr = load_set_model('./models/tflite_model.tflite')
    app.run(host=MLdr.env['APP_HOST'], port=MLdr.env['APP_PORT'], debug=True)
