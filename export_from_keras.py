import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import keras2onnx
import onnx

# load keras model
from keras.applications.vgg16 import VGG16
model = VGG16(include_top=True, weights='imagenet')

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=10)

temp_model_file = 'vgg16_keras.onnx'
onnx.save_model(onnx_model, temp_model_file)
