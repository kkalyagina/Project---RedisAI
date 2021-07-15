from collections import Counter
import os
import random

import albumentations as A
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
import numpy as np

from RedMask.training.dataloader import DataLoader

tf.compat.v1.disable_v2_behavior()
tf.enable_control_flow_v2()

random.seed(42)
# Datasets paths:
CMFD = '/app/data/Datasets/Masks/CMFD'
IMFD = '/app/data/Datasets/Masks/IMFD'
NO_MASKS = '/app/data/Datasets/lfw'
BAD_IMAGES_FILE = '/app/data/bad_images.txt'


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.

    Parameters
    ----------
    session : TF session
        The TensorFlow session to be frozen.
    keep_var_names : list, optional
        A list of variable names that should not be frozen, or None
        to freeze all the variables in the graph. The default is None.
    output_names : list, optional
        Names of the relevant graph outputs. The default is None.
    clear_devices : bool, optional
        Remove the device directives from the graph for better portability.
        The default is True.

    Returns
    -------
    frozen_graph
        The frozen graph definition.

    """

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                                                                            session,
                                                                            input_graph_def,
                                                                            output_names,
                                                                            freeze_var_names)
        return frozen_graph


def get_pathes_and_labels(dataset,
                          cmfd=CMFD, imfd=IMFD,
                          no_masks=NO_MASKS,
                          bad_images_file=BAD_IMAGES_FILE):
    """
    Just gets paths and labels.

    Parameters
    ----------
    dataset: str
        'small' if we want to use small dataset
    cmfd : str
        CMFD dataset path.le "/app/RedMask/utils/training_funcs.py", line 225, in train_model
    imfd : str
        IMFD dataset path.
    no_masks : str
        LFW dataset path.
    bad_images_file : str
        Path to bad_images.txt file.

    Returns
    -------
    img_paths
        Images paths.
    labels
        Images labels.

    """
    # Not so elegant decision...
    if dataset == "small":
        cmfd = '/app/data/Datasets/Masks-small/CMFD'
        imfd = '/app/data/Datasets/Masks-small/IMFD'
        bad_images_file = '/app/data/Datasets/Masks-small/bad_images-small.txt'

    bad_images=[]
    with open(bad_images_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            bad_images.append(no_masks + line[:-1])

    img_pathes = []
    labels = {}
    for phase in (imfd, cmfd):
        folders = os.listdir(phase)
        for folder in folders:
            folder_path = os.path.join(phase, folder)
            img_names = os.listdir(folder_path)
            for img_name in img_names:
                full_path = os.path.join(folder_path, img_name)
                if full_path not in bad_images:
                    img_pathes.append(full_path)
                    if img_name.find('_Mouth_Chin') != -1:
                        labels[full_path] = 2
                    elif img_name.find('_Chin') != -1:
                        labels[full_path] = 3
                    elif img_name.find('_Nose_Mouth') != -1:
                        labels[full_path] = 4
                    else:
                        labels[full_path] = 0

    names = os.listdir(no_masks)
    for name in names:
        full_path = os.path.join(no_masks, name)
        img_pathes.append(full_path)
        labels[full_path] = 1
    return img_pathes, labels


def define_model():
    """
    Creates an empty TF model with a specific architecture.
    Model gets an image with the shape 224х224х3 with variables in the range [0, 1]


    Returns
    -------
    model : TF model
        TensorFlow model.

    """

    baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    headModel = AveragePooling2D(pool_size=(7, 7))(baseModel.output)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(5, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    return model

def train_model(device, dataset):
    """
    This function gets and prepares data, sets augmentations and parameters,
    and then trains the model.

    Parameters
    ----------
    device : string
        device to train on('CPU' or 'GPU').
    dataset : string
        dataset to train on('full' or 'small').

    Returns
    -------
    model : TensorFlow 1.15 model
        The model to work with.

    tflite_model : TFLite model
        The model to work with.

    """
    # Get the data
    img_pathes, labels = get_pathes_and_labels(dataset)
    # compute class weights
    class_weights = Counter(labels.values())
    for k in class_weights.keys():
        class_weights[k] = len(labels) / class_weights[k]
    class_weights = dict(class_weights)

    random.shuffle(img_pathes)
    X_train, X_val = img_pathes[:int(0.9*len(img_pathes))], img_pathes[int(0.9*len(img_pathes)):]

    #Augmentations
    aug = A.Compose([
        A.Resize(224, 224, p=1)
    ])

    batch_size = 8
    train_loader = DataLoader(X_train, labels, batch_size=batch_size, transforms=aug)
    val_loader = DataLoader(X_val, labels, batch_size=batch_size, transforms=A.Resize(224, 224, p=1))

    # Parameters set
    print('\nModel training starts...\n')
    INIT_LR = 1e-30
    EPOCHS = 1
    lr_schedule = ExponentialDecay(
        initial_learning_rate=INIT_LR,
        decay_steps=1000,
        decay_rate=0.1)
    opt = Adam(learning_rate=lr_schedule)
    # Create empty model
    model = define_model()
    # Model training
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])
    with tf.device(device):
        H = model.fit_generator(
                 generator=train_loader,
                 validation_data=val_loader,
                 use_multiprocessing=True,
                 workers=8,
                 epochs=EPOCHS,
                 class_weight=class_weights)
        accuracy = H.history['categorical_accuracy'][0].astype(float)
        print(f"Categorical accuracy: {accuracy}")

    # tflite_model = model_save(model, X_val, labels)
    model = model_save(model, X_val, labels)

    return model, accuracy


def representative_dataset(X_val, labels):
    """
    Assistive function.
    """
    test_loader = DataLoader(X_val, labels, batch_size=1, transforms=A.Resize(224, 224, p=1))
    for i in range(100):
        yield [test_loader[i][0].astype(np.float32)]


def model_save(model, X_val, labels):
    """
    Exports given model to .pb and .tflite formats.
    Saves TF and TFLite models to '/output' folder

    Parameters
    ----------
    model : TF 1.15 model
        Input TF 1.15 model.
    X_val
        Assistive argument.
    labels
        Assistive argument.

    Returns
    -------
    tflite_model : TFLite model
        The TFLite model.
    """
    # Now we need to save the models
    # Freeze session
    tf.enable_control_flow_v2()
    frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(),
                                  output_names=[out.op.name for out in model.outputs])

    # Save to ./model/tf_model.pb
    tf.io.write_graph(frozen_graph, "../output", "frozen_model_1.15.0.pb", as_text=False)
    print('\n TF 1.15 model saved!\n')

    # We need to open frozen graph from file to return it back
    model = open('./data/models/frozen_model_1.15.0.pb', 'rb').read()

    # We have to save model to .h5 format first to save TFLite format because of damn TF1...
    # keras_model='../output/keras_model.h5'
    # model.save(keras_model, X_val, labels)

    return model


def save_tflite(model, X_val, labels):
    """
    Converts given TF model to TFLite model

    Parameters
    ----------
    model : model
        keras model to convert.

    Returns
    -------
    tflite_model :
        TFLite model.

    """
    # Converting a tf.keras model.
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    converter.representative_dataset = representative_dataset(X_val, labels)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Save the model.
    with open('../output/tflite_model_tf_1_15.tflite', 'wb') as f:
        f.write(tflite_model)
        print('\nTFLite model(TF 1.15) saved!\n')

    # Do not need this file anyway
    os.remove('../output/keras_model.h5')

    # return tflite_model
