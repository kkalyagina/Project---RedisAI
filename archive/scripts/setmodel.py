#set model to MLFlow and RedisAI, not used at this moment
from RedMask.modelserve.ModelLoader import ModelLoader

MLdr = ModelLoader()
model_file = open('./models/frozen_model_1.15.0.pb', 'rb').read()
MLdr.upload_model('RedMask_TF', model_file)
model_downloaded = MLdr.download_model('RedMask_TF') #not nesessary
MLdr.set_model("tf_model", 'TF', 'CPU', inputs=['input_1'], outputs=['dense/BiasAdd'])



