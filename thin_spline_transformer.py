import lasagne
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.tensor.nlinalg as la
import numpy as np


class ThinSplineTransformerLayer(lasagne.layers.MergeLayer):
    """
    Spatial transformer layer
    The layer applies an thin spline transformation on the input. The thin spline transform is parameterized by
    The output is interpolated with a bilinear transformation.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    localization_network : a :class:`Layer` instance
        The network that calculates the parameters of the affine
        transformation. See the example for how to initialize to the identity
        transform.
    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  Principal warps: thin-plate splines and the decomposition of deformations.
            Fred L. Bookstein, 1989, IEEE Transactions on Pattern Analysis and Machine Intelligence.
            http://doi.org/10.1109/34.24792

    Examples
    --------
    Here we set up the layer to initially do the identity transform, similarly
    to [1]_. Note that you will want to use a localization with linear output.
    If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
    then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
    t6 move the center position.
    >>> import numpy as np
    >>> import lasagne
    >>> b = np.zeros((2, 3), dtype='float32')
    >>> b[0, 0] = 1
    >>> b[1, 1] = 1
    >>> b = b.flatten()  # identity transform
    >>> W = lasagne.init.Constant(0.0)
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
    ... nonlinearity=None)
    >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
    """
    def __init__(self, incoming, localization_network, downsample_factor=1,
                 num_control_points=16, **kwargs):
        super(ThinSplineTransformerLayer, self).__init__(
            [incoming, localization_network], **kwargs)
        self.downsample_factor = lasagne.utils.as_tuple(downsample_factor, 2)
        self.num_control_points = num_control_points

        input_shp, loc_shp = self.input_shapes

        if loc_shp[-1] != 6 or len(loc_shp) != 2:
            raise ValueError("The localization network must have "
                             "output shape: (batch_size, 6)")
        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s / f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        input, control_points = inputs
        return _transform(control_points, input, self.downsample_factor)


def _transform(dest_points, input, downsample_factor):
    num_batch, num_channels, height, width = input.shape
    num_control_points = dest_points.shape[1]/2
    num_control_points_printed = theano.printing.Print('num_control_points')(num_control_points)

    # grid of (sqrt(num_control_points), sqrt(num_control_points), 1), similar to eq (1) in ref [1]
    source_points = _generate_tps_source_grid(num_control_points)
    sp_shape_printed = theano.printing.Print('source_points')(source_points.shape)
    sp_printed = theano.printing.Print('source_points')(source_points)

    # reshape destination points
    dest_points = T.reshape(dest_points, (num_batch, 2, num_control_points))
    dp_shape_printed = theano.printing.Print('dest_points shape')(dest_points.shape)
    dp_printed = theano.printing.Print('dest_points')(dest_points)



    ## Solve the thin plate spline transform
    # Get number of equations (equal to the number of control points + the bias and x and y translation components)
    num_equations = num_control_points + 3
    num_eq_printed = theano.printing.Print('num_eq')(num_equations)


    # Initialize L matrix
    L = T.zeros((num_equations, num_equations), dtype=theano.config.floatX)
    L_shape_printed = theano.printing.Print('L shape')(L.shape)


    # Create P matrix from [2]
    L = T.set_subtensor(L[0, 3:num_equations], 1.)
    L = T.set_subtensor(L[1:3, 3:num_equations], source_points)
    L = T.set_subtensor(L[3:num_equations, 0], 1.)
    L = T.set_subtensor(L[3:num_equations, 1:3], source_points.T)


    # Calculate U distance for each pair of points in L
    # L, updates = theano.scan(fn=_create_K_matrix,
    #                          outputs_info=L,
    #                          sequences=[T.arange(num_control_points)],
    #                          non_sequences=[T.constant(0), source_points])

    L, updates = theano.scan(fn=_inner_source_U_scan,
                             outputs_info=L,
                             sequences=[T.arange(num_control_points)],
                             non_sequences=[source_points, num_control_points])
    L = L[-1, :, :]
    # L_printed = theano.printing.Print('L')(L.shape)
    L_printed = theano.printing.Print('L')(L)

    # invert the L matrix
    L_inv = la.matrix_inverse(L)
    L_inv_printed = theano.printing.Print('L_inv')(L_inv)


    # Solve
    coefficients = _solve_tps(num_equations, dest_points, L_inv, num_batch)
    # coef_printed = theano.printing.Print('coef_shape')(coefficients.shape)
    coef_printed = theano.printing.Print('coef')(coefficients[:, :, 0])

    # Transformed grid
    out_height = T.cast(height / downsample_factor[0], 'int64')
    out_width = T.cast(width / downsample_factor[1], 'int64')
    orig_grid = _meshgrid(out_height, out_width)
    orig_grid = orig_grid[0:2, :]
    orig_grid = T.tile(orig_grid, (num_batch, 1, 1))
    orig_grid_printed = theano.printing.Print('orig_grid')(orig_grid.shape)

    # Transform each point on the source grid (image_size x image_size)
    transformed_points = _get_transformed_points(orig_grid, source_points,
                                                 coefficients, num_control_points,
                                                 num_batch)
    tp_printed = theano.printing.Print('transformed_points')(transformed_points[:, :, 0])


    # Get out new points
    x_transformed = transformed_points[:, 0].flatten()
    y_transformed = transformed_points[:, 1].flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_transformed, y_transformed,
        out_height, out_width)

    gp_print = theano.printing.Print('test')(input_dim.shape)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output, gp_print


def _U_func(x1, y1, x2, y2):
    """
    Wrapper for _U_func_int which implements the necessary if-else
    """
    return ifelse(T.and_(T.eq(x1, x2), T.eq(y1, y2)),
                  T.cast(T.constant(0), theano.config.floatX),
                  _U_func_int(x1, y1, x2, y2))


def _U_func_int(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * T.log(r_2)


def _get_transformed_points(new_points, source_points, coefficients, num_points, batch_size):
    """
    Calculates the transformed point's value using the provided coefficients
    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2, control_points + 3))
    :param num_points: the number of points
    :return: the x and y coordinates of the transformed point
    """

    # Calculate the squared distance between the new point and the source points
    to_transform = new_points.dimshuffle(0, 'x', 1, 2)  # (batch_size, 1, 2, num_to_transform)
    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))  # (batch_size, num_points, 2, num_to_transform)
    r_2 = T.sum(((stacked_transform - source_points.dimshuffle('x', 1, 0, 'x')) ** 2), axis=2)

    # Calculate the U function for the new point and each source point
    log_r_2 = T.log(r_2)
    distances = T.switch(T.isnan(log_r_2), r_2 * log_r_2, 0.)

    # Add in the coefficients for the affine transform (1, x, and y, corresponding to a_1, a_x, and a_y)
    upper_array = T.concatenate([T.ones((batch_size, 1, new_points.shape[2]), dtype=theano.config.floatX),
                                 new_points], axis=1)
    right_mat = T.concatenate([upper_array, distances], axis=1)

    # Calculate the new value as the dot product
    new_value = T.batched_dot(coefficients, right_mat)
    return new_value


def _get_transformed_point(new_x, new_y, source_points, coefficients, num_points):
    """
    Calculates the transformed point's value using the provided coefficients
    :param new_x: x point to transform
    :param new_y: y point to transform
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (2, control_points + 3))
    :param num_points: the number of points
    :return: the x and y coordinates of the transformed point
    """

    # Calculate the squared distance between the new point and the source points
    to_transform = T.stack([new_x, new_y])
    stacked_transform = T.tile(to_transform, (num_points, 1))
    r_2 = T.sum(((stacked_transform.T - source_points) ** 2), axis=0)

    # Calculate the U function for the new point and each source point
    distances = r_2 * T.log(r_2)

    # Add in the coefficients for the affine transform (1, x, and y, corresponding to a_1, a_x, and a_y)
    upper_array = T.stack([T.constant(1.), new_x, new_y])
    right_mat = T.concatenate([upper_array, distances])

    # Calculate the new value as the dot product
    new_value = T.dot(coefficients, right_mat)
    return new_value


def _solve_tps(num_equations, dest_points, L_inv, batch_size):
    coefficients = T.zeros((batch_size, 2, num_equations), dtype=theano.config.floatX)
    coefficients, updates = theano.scan(fn=_solve_inner_scan,
                                        outputs_info=coefficients,
                                        sequences=[T.arange(num_equations)],
                                        non_sequences=[T.constant(0), dest_points, num_equations, L_inv])
    coefficients = coefficients[-1, :, :, :]
    coefficients, updates = theano.scan(fn=_solve_inner_scan,
                                        outputs_info=coefficients,
                                        sequences=[T.arange(num_equations)],
                                        non_sequences=[T.constant(1), dest_points, num_equations, L_inv])
    return coefficients[-1, :, :, :]


def _solve_inner_scan(eq_1, coefficients, variable, dest_points, num_equations, L_inv):
    coefficients, updates = theano.scan(fn=_inner_solve,
                                        outputs_info=coefficients,
                                        sequences=[T.arange(3, num_equations)],
                                        non_sequences=[eq_1, variable, dest_points, L_inv])

    return coefficients[-1, :, :, :]


def _inner_solve(eq_2, coefficients, eq_1, variable, dest_points, L_inv):
    return T.set_subtensor(coefficients[:, variable, eq_1],
                           coefficients[:, variable, eq_1] +
                           L_inv[eq_1, eq_2] * dest_points[:, variable, eq_2 - 3])


def _inner_source_U_scan(curr_point, L, source_points, num_control_points):
    """
    Performs the inner loop for the nested for loop which calculates the U values
    """
    L, updates = theano.scan(fn=_create_K_matrix,
                             outputs_info=L,
                             sequences=[T.arange(num_control_points)],
                             non_sequences=[curr_point, source_points])
    return L[-1, :, :]


def _create_K_matrix(point_1, L, point_2, source_points):
    temp_val = _U_func(source_points[0, point_1], source_points[1, point_1],
                       source_points[0, point_2], source_points[1, point_2])
    L = T.set_subtensor(L[point_1 + 3, point_2 + 3], temp_val)
    L = T.set_subtensor(L[point_2 + 3, point_1 + 3], temp_val)
    return L


def _interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def _generate_tps_source_grid(num_control_points):
    """
    Generates source and destination grid coordinates from the control points
    :param num_control_points: number of control points
    :return: a vertical stack of the source grid
    """

    # Get grid size
    grid_size = T.cast(T.sqrt(num_control_points), 'int64')

    # generate mesh grid for source points
    x_source = T.dot(T.ones((grid_size, 1)),
                     _linspace(-1.0, 1.0, grid_size).dimshuffle('x', 0))
    y_source = T.dot(_linspace(-1.0, 1.0, grid_size).dimshuffle(0, 'x'),
                     T.ones((1, grid_size)))

    # flatten
    x_source_flat = x_source.reshape((1, -1))
    y_source_flat = y_source.reshape((1, -1))

    # Concatenate
    source_points = T.concatenate([x_source_flat, y_source_flat], axis=0)

    # Tile
    # source_points = T.tile(source_points, (batch_size, 1, 1))

    return source_points
