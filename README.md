# Exporting scripts for ONNX and ailia SDK

## Pytorch
A model learned with Pytorch can be loaded by outputting it to ONNX format using torch.onnx.export, then using the included script tools/onnx/onnx2prototxt.py to convert it to a "prototxt" format that then can be loaded into AILIA.

Export from Pytorch to ONNX

    torch.onnx.export(vgg16, x, 'vgg16_pytorch.onnx', verbose=True)

Creation of the "prototxt" file

    python3 onnx2prototxt.py vgg16.onnx

##	Keras
A model learned with Keras can be loaded by outputting it to ONNX format using keras2onnx, then using the included script tools/onnx/onnx2prototxt.py to convert it to a "prototxt" file that then can be loaded into AILIA.

Installation of keras2onnx

    pip3 install onnx
    pip3 install keras2onnx

Exporting from Keras to ONNX

    onnx_model = keras2onnx.convert_keras(model, model.name, target_opset=10)
    temp_model_file = 'vgg16.onnx'
    onnx.save_model(onnx_model, temp_model_file)

Creation of the prototxt file

    python3 onnx2prototxt.py vgg16.onnx

## Chainer
A model learned with Chainer can be loaded by outputting it to ONNX format using onnx-chainer, then using the included script tools/onnx/onnx2prototxt.py to convert it to a "prototxt" file that then can be loaded into AILIA.

Installation of onnx-chainer

    pip3 install onnx
    pip3 install onnx-chainer

Exporting from Chainer to ONNX

    onnx_chainer.export(model, x, filename='vgg16.onnx', opset_version=10)

Creation of the prototxt file

    python3 onnx2prototxt.py vgg16.onnx

##	Tensorflow
A model learned with Tensorflow can be loaded by outputting it to ONNX format using tf2onnx, then using the included script tools/onnx/onnx2prototxt.py to convert it to a "prototxt" file that then can be loaded into AILIA.

Installation of tf2onnx

    pip3 install tf2onnx

Exporting from Tensorflow to ONNX

    import tf2onnx

    frozen_graph_def = 
    tf.graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),["vgg16/predictions/Softmax"])

    graph1 = tf.Graph()
    with graph1.as_default():
        tf.import_graph_def(frozen_graph_def)
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph1, input_names=["import/block1_conv1/kernel:0"], output_names=["import/vgg16/predictions/Softmax:0"],opset=10)
        model_proto = onnx_graph.make_model("vgg16")
        with open("vgg16.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())```

Creation of the prototxt file

    python3 onnx2prototxt.py vgg16.onnx


