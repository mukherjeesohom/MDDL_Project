import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from model import *

def compute_params_flops(model, n_channels=3, n_size=32):
    dummy_input = [tf.TensorSpec(shape=(1, n_size, n_size, n_channels))] 
    forward_pass = tf.function(model.call, input_signature=dummy_input)
    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops // 2
    print('MACs: {:,}'.format(flops))

if __name__ =="__main__":
    # Create a new model instance; choose from: 'resnet32_global', 'resnet32'
    model_name = 'resnet32_global'
    model = ResNet(model_name, 10)
    compute_params_flops(model)

    model.build((None, 32, 32, 3))
    model.summary()