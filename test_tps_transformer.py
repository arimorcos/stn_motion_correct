import lasagne
from thin_spline_transformer import ThinSplineTransformerLayer
from theano.tensor import constant
from theano import config
import numpy as np


if __name__ == "__main__":

    batch_size = 2

    # create transformer with fixed input size
    l_in = lasagne.layers.InputLayer((batch_size, 3, 28, 28))
    l_loc = lasagne.layers.DenseLayer(l_in, num_units=6)
    l_trans = ThinSplineTransformerLayer(l_in, l_loc, num_control_points=16)

    # Create inputs
    inputs = np.random.normal(0, 1, l_in.shape).astype(config.floatX)

    # Create control points
    num_control_points = 16
    grid_size = np.sqrt(num_control_points)
    x_control_t, y_control_t = np.meshgrid(np.linspace(-1, 1, grid_size),
                                           np.linspace(-1, 1, grid_size))
    x_offset = 0
    y_offset = 0
    control_points = np.vstack((x_control_t.flatten() + x_offset,
                                y_control_t.flatten() + y_offset)).T.flatten()

    # Get outputs
    outputs = l_trans.get_output_for([constant(inputs), constant(control_points)]).eval()
