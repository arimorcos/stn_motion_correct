import lasagne
from thin_spline_transformer import ThinSplineTransformerLayer
from spatial_transformer_affine import TransformerLayer
from theano.tensor import constant
import theano
from theano import config
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import lena
import time


if __name__ == "__main__":

    batch_size = 3
    num_control_points = 16

    # create transformer with fixed input size
    l_in = lasagne.layers.InputLayer((batch_size, 3, 512, 512))
    l_loc = lasagne.layers.DenseLayer(l_in, num_units=2*num_control_points)
    # l_loc = lasagne.layers.DenseLayer(l_in, num_units=6)
    l_trans = ThinSplineTransformerLayer(l_in, l_loc, num_control_points=num_control_points,
                                         downsample_factor=1)
    # l_trans = TransformerLayer(l_in, l_loc)

    # Create inputs
    # inputs = np.random.normal(0, 1, l_in.shape).astype(config.floatX)
    # inputs = np.arange(np.prod(l_in.shape)).reshape(l_in.shape)
    # inputs = np.tile(np.outer(np.arange(l_in.shape[2]), np.arange(l_in.shape[3])),
    #                  (l_in.shape[0], l_in.shape[1], 1, 1))
    # inputs = np.tile(lena()[::18, ::18], (l_in.shape[0], l_in.shape[1], 1, 1))
    inputs = np.tile(lena(), (l_in.shape[0], l_in.shape[1], 1, 1))

    # Create control points
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(np.linspace(-1, 1, grid_size),
                                                     np.linspace(-1, 1, grid_size))
    # x_offset = np.random.normal(0, 0.3, x_control_source.size)
    # y_offset = np.random.normal(0, 0.3, x_control_source.size)
    x_offset = 0.5
    y_offset = 0
    x_control_dest = x_control_source.flatten() + x_offset
    y_control_dest = y_control_source.flatten() + y_offset
    dest_points = np.vstack((x_control_dest, y_control_dest)).flatten()

    dest_points = np.tile(dest_points, (batch_size, 1)).astype(config.floatX)

    # Get outputs
    # outputs, printed = l_trans.get_output_for([constant(inputs), constant(dest_points)])
    outputs = l_trans.get_output_for([constant(inputs), constant(dest_points)])
    # thetas = np.tile([1, 0, 0, 0, 1, 0], (batch_size, 1))
    # print thetas

    # outputs, printed = l_trans.get_output_for([constant(inputs), constant(thetas)])
    # outputs = outputs.eval()
    comp_start = time.time()
    func = theano.function([], outputs)
    print "Compilation time: {}".format(time.time() - comp_start)

    # printed = printed.eval()
    start = time.time()
    outputs = func()
    print "Run time: {}".format(time.time() - start)

    # print np.allclose(outputs, inputs)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.imshow(inputs[0, 0, :, :], cmap='gray')
    ax = fig.add_subplot(122)
    ax.imshow(outputs[0, 0, :, :], cmap='gray')
    plt.show()


