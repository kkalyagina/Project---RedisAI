import argparse
import tensorflow as tf

from RedMask.modelserve.ModelLoader import ModelLoader
from RedMask.training.training_funcs import train_model

tf.enable_control_flow_v2()


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
    try:
        # Check if GPU avalible
        print("\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        # By default using CPU for training
        gpu_available = tf.config.experimental.list_physical_devices('GPU')
        device = "/device:GPU:0" if gpu_available else "/device:CPU:0"
    # If we have some problems with GPU on the host machine:
    except:
        device = "/device:CPU:0"

    return device


if __name__ == "__main__":
    # do that
    tf.compat.v1.disable_v2_behavior()
    # Some usefil parameters
    parser = argparse.ArgumentParser()
    # Dataset can be 'full'  for the entire dataset, or 'small' for the small dataset
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset to train on. It can be small or full',
                        default='small')
    args = parser.parse_args()
    # COMMENT FOLLOWING TO BYPASS TRAINING
    # Define a device
    device = define_device()
    model, accuracy = train_model(device, args.dataset)

    # UNCOMMENT IF YOU DO NOT WANT TO TRAIN NEW TF FROZEN GRAPH
    #model = open('./data/models/frozen_model_1.15.0.pb', 'rb').read()
    #model = open('/home/kos/EPAM/students_redisai/data/models/frozen_model_1.15.0.pb', 'rb').read()
    #accuracy = 0.22

    # COMMENT FOLLOWING TO CANCEL TRAINING BYPASS
    # model = open('./data/models/tflite_model.tflite', 'rb').read()
    # model = open('./data/models/frozen_model_1.15.0.pb', 'rb').read()
    # accuracy = 0.77
    # device='bypass'

    # set tags, metrics and 
    metrics = {'accuracy': accuracy}
    tags = {'device': device, 'dataset': args.dataset}
    # connect to servers, upload and update model
    MLdr = ModelLoader()
    experiment_id = MLdr.upload_model(model_name='RedMask_TF_2',
                                      model=model,
                                      metrics=metrics,
                                      tags=tags)
    MLdr.updatemodel('RedMask_TF_2', 'TF', 'CPU',
                     inputs=['input_1'], outputs=['dense/BiasAdd']
                     )
