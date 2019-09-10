import numpy as np

import chainer
import chainercv.links as C
import onnx_chainer

model = C.VGG16(pretrained_model='imagenet')
x = np.zeros((1, 3, 224, 224), dtype=np.float32)
onnx_chainer.export(model, x, filename='vgg16_chainer.onnx', opset_version=10)
