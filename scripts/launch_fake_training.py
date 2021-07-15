import argparse

from RedMask.modelserve.ModelLoader import ModelLoader
#from RedMask.training.training_funcs import train_model

def define_device():
    """
    Define avalible device. Training will be quite fast 
    if the host has a compatible GPU.

    Returns
    -------
    device : str
        String that contains GPU or CPU instruction
        for the training.

    """
    #Check if GPU avalible
    print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    #By default using CPU for training
    gpu_available = tf.config.experimental.list_physical_devices('GPU')
    device = "/device:GPU:0" if gpu_available else "/device:CPU:0"

    return device

#model, tf_model = train_model(device)


if __name__ == "__main__":
    #Some usefil parameters
    parser = argparse.ArgumentParser()
    #Dataset can be 'full'  for the entire dataset, or 'small' for the small dataset
    parser.add_argument('--dataset',
                        type = str,
                        help='dataset to train on. It can be small or full',
                        default='small')
    args = parser.parse_args()
    #COMMENT FOLLOWING TO BYPASS TRAINING
    # Define a device
    #device = define_device()
    #train model - commented until ran on suitable machine
    #model, accuracy = train_model(device, args.dataset)

    #COMMENT FOLLOWING TO CANCEL TRAINING BYPASS
    model = open('./data/models/tflite_model.tflite', 'rb').read()
    accuracy = 0.77
    device='bypass'

    #set tags, metrics and 
    metrics = {'accuracy': accuracy}
    tags = {'device' : device, 'dataset' : args.dataset}
    #connect to servers, upload and update model
    MLdr = ModelLoader()
    experiment_id = MLdr.upload_model(model_name='RedMask_TFLite', model=model, metrics=metrics, tags=tags)
    MLdr.updatemodel('RedMask_TFLite', 'TFLite', 'CPU', inputs=['input_1'], outputs=['dense/BiasAdd'])


