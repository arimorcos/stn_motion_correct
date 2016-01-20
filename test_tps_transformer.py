import lasagne
from thin_spline_transformer import ThinSplineTransformerLayer
from spatial_transformer_affine import TransformerLayer
from theano.tensor import constant
from theano import config
import numpy as np


if __name__ == "__main__":

    batch_size = 2

    # create transformer with fixed input size
    l_in = lasagne.layers.InputLayer((batch_size, 3, 28, 28))
    l_loc = lasagne.layers.DenseLayer(l_in, num_units=32)
    l_trans = ThinSplineTransformerLayer(l_in, l_loc, num_control_points=16)
    # l_trans = TransformerLayer(l_in, l_loc)

    # Create inputs
    # inputs = np.random.normal(0, 1, l_in.shape).astype(config.floatX)
    inputs = np.arange(np.prod(l_in.shape)).reshape(l_in.shape)

    # Create control points
    num_control_points = 16
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(np.linspace(-1, 1, grid_size),
                                                     np.linspace(-1, 1, grid_size))
    x_offset = 0.5
    y_offset = 0
    x_control_dest = x_control_source.flatten() + x_offset
    y_control_dest = y_control_source.flatten() + y_offset
    dest_points = np.vstack((x_control_dest, y_control_dest)).flatten()

    # Get outputs
    outputs, printed = l_trans.get_output_for([constant(inputs), constant(dest_points)])
    # theta = np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 2).transpose(2, 0, 1)
    # theta = np.tile(theta, (batch_size, 1, 1))
    # theta = np.array([[1, 0, 0], [0, 1, 0]]).flatten()
    # theta = np.tile(theta, (batch_size, 1))
    # thetas = np.tile([1, 0, 0, 0, 1, 0], (batch_size, 1))
    # print thetas

    # outputs, printed = l_trans.get_output_for([constant(inputs), constant(thetas)])
    outputs = outputs.eval()
    printed = printed.eval()

    print np.allclose(outputs, inputs)
    print outputs.shape
    print inputs.shape
