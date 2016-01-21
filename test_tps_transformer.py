import lasagne
from thin_spline_transformer import ThinSplineTransformerLayer
from spatial_transformer_affine import TransformerLayer
from theano.tensor import constant
from theano import config
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    batch_size = 5

    # create transformer with fixed input size
    l_in = lasagne.layers.InputLayer((batch_size, 3, 28, 28))
    l_loc = lasagne.layers.DenseLayer(l_in, num_units=6)
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
    x_offset = 0.25
    y_offset = 1
    x_control_dest = x_control_source.flatten() + x_offset
    y_control_dest = y_control_source.flatten() + y_offset
    dest_points = np.vstack((x_control_dest, y_control_dest)).flatten()

    dest_points = np.tile(dest_points, (batch_size, 1)).astype(config.floatX)

    # Get outputs
    outputs, printed = l_trans.get_output_for([constant(inputs), constant(dest_points)])
    # thetas = np.tile([1, 0, 0, 0, 1, 0], (batch_size, 1))
    # print thetas

    # outputs, printed = l_trans.get_output_for([constant(inputs), constant(thetas)])
    outputs = outputs.eval()
    printed = printed.eval()

    print np.allclose(outputs, inputs)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.imshow(inputs[0, 0, :, :], cmap='gray')
    ax = fig.add_subplot(122)
    ax.imshow(outputs[0, 0, :, :], cmap='gray')
    plt.show()


