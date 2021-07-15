# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from mtcnn_custom import __MTCNN
import tensorflow as tf


from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.savefig('../images/output/image_with_face_boxes.png')
	pyplot.show()


print('Saving models...')

#

#for the TF2:
def make_graphs():
    """
    Save three models used in this task at /graphs folder in the .pb format. 
    """
    #models from original MTCNN package
    detector = MTCNN()
    models = [detector._pnet, detector._rnet, detector._onet]
    for i,model in enumerate(models):
    	full_model = tf.function(lambda x: model(x))
    	full_model = full_model.get_concrete_function(
    		         tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    
    	frozen_func = convert_variables_to_constants_v2(full_model)
    	frozen_func.graph.as_graph_def()
    	
    	tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    		          logdir=".",
    		          name=f'./graphs/graph_{str(i)}.pb',
    		          as_text=False)
    print('Models saved!')




#check if models exists
if (os.path.isfile('graphs/graph_0.pb') == False) | \
    (os.path.isfile('graphs/graph_1.pb') == False)| \
    (os.path.isfile('graphs/graph_2.pb') == False):
    make_graphs()        
    print('Need to make graphs from TensorFlow models...')
#Filename of the test image. The default name is './src/input/image.jpeg'.
filename = '../images/input/image.jpeg'
# load image from file
pixels = pyplot.imread(filename)
# create the CUSTOM detector, using default weights
detector = __MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
#print(faces)
draw_image_with_boxes(filename, faces)

#For the ONNX
#tf.saved_model.save(tf.keras.models.Model(detector._pnet), './src/output/tensorflow_pnet')
#tf.saved_model.save(tf.keras.models.Model(detector._rnet), './src/output/tensorflow_rnet')
#tf.saved_model.save(tf.keras.models.Model(detector._onet), './src/output/tensorflow_onet')
#print('Models saved. Now you can convert them to onnx format.')

