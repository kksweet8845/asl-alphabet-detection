import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np




def freeze_graph(model, frozen_out_path, frozen_graph_filename):

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(tf.expand_dims(x, axis=0)))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    # Get Frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-"*60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print('-' * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("froze model outputs: ")
    print(frozen_func.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=frozen_out_path,
                      name=f"{frozen_graph_filename}.pb",
                      as_text=False)

    # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    #                   logdir=frozen_out_path,
    #                   name=f"{frozen_graph_filename}.pbtxt",
    #                   ax_text=True)