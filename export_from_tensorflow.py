from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf

# model
base_model = VGG16(weights='imagenet',
                   include_top=True,
                   pooling='avg')

x = tf.placeholder(tf.float32, [1, 224, 224, 3], name="vgg16_input")
y = tf.placeholder(tf.float32, [1, 1, 1, 1000], name="vgg16_output")

y = base_model(x)

base_model.summary()

# trainable & uninitialized variables
uninitialized_variables = [v for v in tf.global_variables() \
    if not hasattr(v, '_keras_initialized') or not v._keras_initialized]

# initialization
sess = K.get_session()
sess.run(tf.variables_initializer(uninitialized_variables))

frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),["vgg16/predictions/Softmax"])

import tf2onnx
from tf2onnx.optimizer.transpose_optimizer import TransposeOptimizer

graph1 = tf.Graph()
with graph1.as_default():
    tf.import_graph_def(frozen_graph_def)
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph1, input_names=["import/block1_conv1/kernel:0"], output_names=["import/vgg16/predictions/Softmax:0"],opset=10)
    optimizer = TransposeOptimizer()
    opt_model_proto = optimizer.optimize(onnx_graph)
    model_proto = onnx_graph.make_model("vgg16_tensorflow")
    with open("vgg16_tensorflow.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
