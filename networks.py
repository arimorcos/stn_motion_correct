import numpy as np
import theano
import lasagne
import theano.tensor as T
import re
import os
import cPickle
import sys
import warnings
import logging
from batch_norm import batch_norm
from tps_helper import apply_tps_transform
from abc import ABCMeta, abstractmethod


class generic_stn:
    """
    Class to instantiate a spatial transformer network
    """
    __metaclass__ = ABCMeta

    def __init__(self, batch_size=32, alpha=0.0005, max_norm=5., log_dir=None, save_every=10,
                 dropout_frac=0.5, initialization='glorot_uniform'):
        """
        :param batch_size: batch_size for the network (pre-specifying allows for theano optimizations)
        :param alpha: initial learning rate
        :param max_norm: maximum norm on the gradient
        :param log_dir: where should the log directory be
        :param save_every: how often to save parameters
        :param dropout_frac: Fraction for dropout
        :param initialization: Which initialization regime to use. Default is 'glorot_uniform'. Options are {
            'glorot_uniform', 'glorot_normal', 'orthogonal', 'he_normal', 'he_uniform'}
        """

        # Initialize parameters
        self.batch_size = batch_size
        self.alpha = alpha
        self.max_norm = max_norm
        self.log_dir = log_dir
        self.initialization = initialization
        self.dropout_frac = dropout_frac
        self.curr_epoch = 0
        self.save_every = save_every

        # Create the graph
        # self.create_network_graph(batch_size=self.batch_size)
        self.create_simple_network_graph(batch_size=self.batch_size)

        # Create relevant inputs
        self.create_inputs()

        # Initialize the process and cost functions
        self.initialize_process()
        self.initialize_cost()

        # Initialize adam
        self.initialize_adam()

    def initialize_adam(self):
        """
        Initializes the adam training function
        """

        # Get the parameters to train
        self.params_to_train = lasagne.layers.get_all_params(self.transformer_graph,
                                                             trainable=True)

        # Get gradients
        self.all_gradients = T.grad(self.cost, self.params_to_train)

        # Add gradient normalization
        updates = lasagne.updates.total_norm_constraint(self.all_gradients,
                                                              max_norm=self.max_norm)

        # Create learning rate
        self.shared_lr = theano.shared(lasagne.utils.floatX(self.alpha))

        # Create adam function
        updates = lasagne.updates.adam(updates,
                                       self.params_to_train,
                                       learning_rate=self.shared_lr)

        # Create train function
        self.train_adam_helper = theano.function([self.input_batch, self.ref_imgs],
                                          self.cost,
                                          updates=updates)

    def train_adam(self, input_batch, ref_imgs):

        # Save parameters
        if self.curr_epoch % self.save_every == 0:
            self.save_parameters()

        # Train network
        cost = self.train_adam_helper(input_batch, ref_imgs)

        # Add to log
        self.logger.info(
            "Batch: {:06d} | Cost: {:.9f}".format(self.curr_epoch, cost.tolist())
        )

        # Increment current epoch
        self.curr_epoch += 1

        return cost

    def create_inputs(self):
        """
        Initializes the tensors for the input images and reference images
        """
        # Create input tensor
        self.input_batch = T.tensor4('input_batch', dtype=theano.config.floatX)

        # Create reference tensor (batch_size, height, width)
        self.ref_imgs = T.tensor3('ref_imgs', dtype=theano.config.floatX)

    def create_network_graph(self, batch_size=32, should_batch_norm=False):
        """
        Builds a spatial transformer network
        :param should_batch_norm: Whether or not to use batch normalization
        :param batch_size: batch_size for optimization
        :return:
        """

        # Set initialization scheme
        W_ini = lasagne.init.GlorotUniform()

        # Input layer with size (batch_size, num_channels, height, width).
        # In our case, each channel will represent the image to change and the reference image.
        input_layer = lasagne.layers.InputLayer((batch_size, 2, 512, 512))
        if should_batch_norm:
            input_layer = batch_norm(input_layer)

        # convolutions
        conv_layer_1 = lasagne.layers.Conv2DLayer(input_layer, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_1', W=W_ini)
        if should_batch_norm:
            conv_layer_1 = batch_norm(conv_layer_1)
        conv_layer_2 = lasagne.layers.Conv2DLayer(conv_layer_1, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_2', W=W_ini)
        if should_batch_norm:
            conv_layer_2 = batch_norm(conv_layer_2)

        # pool
        pool_layer_1 = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(2, 2), name='pool_1')
        if should_batch_norm:
            pool_layer_1 = batch_norm(pool_layer_1)

        # more convolutions
        conv_layer_3 = lasagne.layers.Conv2DLayer(pool_layer_1, num_filters=64, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_3', W=W_ini)
        if should_batch_norm:
            conv_layer_3 = batch_norm(conv_layer_3)
        conv_layer_4 = lasagne.layers.Conv2DLayer(conv_layer_3, num_filters=64, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_4', W=W_ini)
        if should_batch_norm:
            conv_layer_4 = batch_norm(conv_layer_4)

        # pool
        pool_layer_2 = lasagne.layers.MaxPool2DLayer(conv_layer_4, pool_size=(2, 2), name='pool_2')
        if should_batch_norm:
            pool_layer_2 = batch_norm(pool_layer_2)

        # more convolutions
        conv_layer_5 = lasagne.layers.Conv2DLayer(pool_layer_2, num_filters=128, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_5', W=W_ini)
        if should_batch_norm:
            conv_layer_5 = batch_norm(conv_layer_5)
        conv_layer_6 = lasagne.layers.Conv2DLayer(conv_layer_5, num_filters=128, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_6', W=W_ini)
        if should_batch_norm:
            conv_layer_6 = batch_norm(conv_layer_6)

        # pool
        pool_layer_3 = lasagne.layers.MaxPool2DLayer(conv_layer_6, pool_size=(2, 2), name='pool_3')
        if should_batch_norm:
            pool_layer_3 = batch_norm(pool_layer_3)

        # Dense layers
        dense_layer_1 = lasagne.layers.DenseLayer(pool_layer_3, num_units=128, W=W_ini,
                                                  name='dense_1')
        if should_batch_norm:
            dense_layer_1 = batch_norm(dense_layer_1)

        # Initialize affine transform to identity
        b = np.zeros((2, 3), dtype=theano.config.floatX)
        b[0, 0] = 1
        b[1, 1] = 1

        # Final affine layer
        affine_layer = lasagne.layers.DenseLayer(dense_layer_1, num_units=6, W=lasagne.init.Constant(0.0),
                                                 b=b.flatten(), nonlinearity=lasagne.nonlinearities.identity,
                                                 name='affine')

        # Finally, create the transformer network
        transformer = lasagne.layers.TransformerLayer(incoming=input_layer,
                                                      localization_network=affine_layer,
                                                      downsample_factor=1)

        # Slice out the first channel
        transformer = lasagne.layers.SliceLayer(transformer, indices=0, axis=1)

        # Return
        self.transformer_graph = transformer

    def get_param_values(self):
        """
        Returns list of all the parameter values
        :return: List of numpy arrays containing all parameter values
        """

        return lasagne.layers.get_all_param_values(self.transformer_graph)

    def save_parameters(self):
        """def set_parameters(self):
        Saves the parameters to a new file in the current log folder
        """

        if self.log_dir is None:
            raise AttributeError("No log folder specified")

        # Create log file
        log_file = os.path.join(self.log_dir, "Epoch_{:04d}_weights.pkl".format(self.curr_epoch))

        with open(log_file, 'wb') as f:
            param_values = self.get_param_values()
            cPickle.dump(param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def save_model(self, save_file):
        """
        Saves the entire network model
        :param save_file: path to file to save
        :return: None
        """

        # Get old recursion limit and set recursion limit to very high
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(int(1e5))

        # save the model
        with open(save_file, mode='wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # Reset old recursion limit
        sys.setrecursionlimit(old_recursion_limit)

    def set_parameters(self, param_file=None, epoch=None):
        """
        Loads parameters from the specified file and sets them
        :param param_file: file to load parameters from. If specified, loads a specific file.
        :param epoch: Epoch to load from. If specified, loads parameters from a given epoch in the current log
        folder. If both param_file and epoch provided, param_file will be used.
        :return: None
        """

        # Get load file
        if param_file:
            load_file = param_file
            if epoch:
                warnings.warn('Both param_file and epoch provided. Using param_file...')
            if not os.path.isfile(load_file):
                raise IOError('File {} not found. No parameters have been set'.format(load_file))
        elif epoch is not None:
            load_file = os.path.join(self.log_dir, 'Epoch_{:04d}_weights.pkl'.format(epoch))
            if not os.path.isfile(load_file):
                raise IOError('File corresponding to epoch {}: \"Epoch_{:04d}_weights.pkl\" not found.'.format(epoch,
                                                                                                           epoch))
        else:
            raise ValueError('Must provide param_file or epoch. No parameters have been set.')

        # load parameters
        with open(load_file, 'rb') as f:
            loaded_params = cPickle.load(f)
            lasagne.layers.set_all_param_values(self.transformer_graph, loaded_params)

    def set_log_dir(self, log_dir):
        """
        Sets the current log directory
        :param log_dir: path to log directory
        :return: None
        """
        self.log_dir = log_dir

        if os.path.isdir(log_dir):
            # Get file list
            file_list = os.listdir(log_dir)

            # Subset to epoch files
            epoch_string = 'Epoch_\d{4}_weights'
            epoch_files = [item for item in file_list if re.search(epoch_string, item)]

            # Get most up to date epoch and ask if files are present
            if epoch_files:

                # Get list of epoch strings
                epoch_nums = []
                for x in epoch_files:
                    match = re.search('((?<=Epoch\_)(\d{4})(?=\_weights))', x)
                    if match:
                        epoch_nums.append(int(match.group()))

                # get maximum epoch
                max_epoch = max(epoch_nums)

                # ask user
                answer = raw_input("Epoch files already exist. Maximum epoch is {}. Reset y/n?".format(max_epoch))
                if answer == 'y':
                    # Delete files
                    [os.remove(os.path.join(log_dir, x)) for x in epoch_files]
                elif answer == 'n':
                    # Set current epoch to max_epoch + 1
                    self.curr_epoch = max_epoch + 1
                else:
                    raise BaseException("Cannot parse answer.")

        else:
            os.mkdir(log_dir)

        # Create log file
        self.create_log_file()

    def get_log_dir(self):
        """
        :return: Path to current log directory
        """
        if self.log_dir:
            return self.log_dir
        else:
            print "No log directory set."

    def create_log_file(self):
        """
        Creates a log file
        """

        logger = logging.getLogger('')
        logger.handlers = []
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(
                os.path.join(self.log_dir, "results.log"), mode='wb'
        )
        file_handler.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Store
        self.logger = logger

    @abstractmethod
    def create_simple_network_graph(self, batch_size):
        pass


class stn_affine(generic_stn):
    """
    Class to instantiate a spatial transformer network
    """

    def initialize_cost(self):
        """
        Initializes the cost function
        """

        # get the transformed images
        predictions = lasagne.layers.get_output(self.transformer_graph, self.input_batch)

        # add in the cost (mse)
        self.cost = lasagne.objectives.squared_error(predictions, self.ref_imgs).mean()

        #### add in the cost (pixel weighted mse)
        # squared_error = lasagne.objectives.squared_error(predictions, self.ref_imgs)  # get squared error
        #
        # # Manipulate each ref image to be sum to 1
        # ref_shape = self.ref_imgs.shape  # Get shape for convenience
        # ref_img_reshape = T.reshape(self.ref_imgs, newshape=(ref_shape[0], ref_shape[1]*ref_shape[2]),
        #                             ndim=2)  # reshape to batch_size x num_pixels
        # # self.get_shape_1 = theano.function([self.ref_imgs], ref_img_reshape.shape)
        # ref_img_max = ref_img_reshape.max(axis=1).dimshuffle((0, 'x'))  # Get max for each image and create
        #                                                                 # broadcastable dimension
        # ref_img_min = ref_img_reshape.min(axis=1).dimshuffle((0, 'x'))  # get min for each image
        # # self.get_max_min_shape = theano.function([self.ref_imgs], [ref_img_max.shape, ref_img_min.shape])
        # ref_img_norm = (ref_img_reshape - ref_img_min) / (ref_img_max - ref_img_min)  # norm each image between 0 and 1
        # # self.get_norm_shape = theano.function([self.ref_imgs], ref_img_norm.shape)
        #
        # # self.get_ref_norm = theano.function([self.ref_imgs], ref_img_norm)
        # #
        # # self.get_norm_max_min = theano.function([self.ref_imgs], [ref_img_norm.max(axis=1), ref_img_norm.min(axis=1)])
        #
        # ref_img_partition = ref_img_norm / T.sum(ref_img_norm, axis=1).dimshuffle((0, 'x'))  # Partition so it sums to 1
        # # self.get_shape_2 = theano.function([self.ref_imgs], ref_img_partition.shape)
        # # norm_sum = T.sum(ref_img_norm, axis=1)
        # # part_sum = T.sum(ref_img_partition, axis=1)
        # # self.new_sum = theano.function([self.ref_imgs], [norm_sum, part_sum])
        # ref_img_partition = T.reshape(ref_img_partition, self.ref_imgs.shape)  # return to original shape
        #
        # # Aggregate using the normalized ref image as the weights
        # self.cost = 100000*(squared_error * ref_img_partition).mean()

        # create function
        self.get_cost = theano.function([self.input_batch, self.ref_imgs], self.cost)

    def initialize_process(self):
        """
        Initializes the process function
        """

        # Create symbolic output
        output = lasagne.layers.get_output(self.transformer_graph, self.input_batch, deterministic=True)

        # Create theano function
        self.process = theano.function([self.input_batch], output)

    def create_simple_network_graph(self, batch_size=32):
        """
        Builds a spatial transformer network
        :param batch_size: batch_size for optimization
        :return:
        """

        # Set initialization scheme
        if self.initialization == 'glorot_uniform':
            W_ini = lasagne.init.GlorotUniform('relu')
        elif self.initialization == 'glorot':
            W_ini = lasagne.init.Glorot('relu')
        elif self.initialization == 'glorot_normal':
            W_ini = lasagne.init.GlorotNormal('relu')
        elif self.initialization == 'he':
            W_ini = lasagne.init.He('relu')
        elif self.initialization == 'he_normal':
            W_ini = lasagne.init.HeNormal('relu')
        elif self.initialization == 'he_uniform':
            W_ini = lasagne.init.HeUniform('relu')
        elif self.initialization == 'orthogonal':
            W_ini = lasagne.init.Orthogonal('relu')
        else:
            raise AttributeError('Initialization string {} not understood'.format(self.initialization))

        # Input layer with size (batch_size, num_channels, height, width).
        # In our case, each channel will represent the image to change and the reference image.
        input_layer = lasagne.layers.InputLayer((batch_size, 2, 512, 512))

        # convolutions
        conv_layer_1 = lasagne.layers.Conv2DLayer(input_layer, num_filters=16, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_1', W=W_ini)
        conv_layer_2 = lasagne.layers.Conv2DLayer(conv_layer_1, num_filters=16, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_2', W=W_ini)

        # pool
        pool_layer_1 = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(2, 2), name='pool_1')

        # convolutions
        conv_layer_3 = lasagne.layers.Conv2DLayer(pool_layer_1, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_3', W=W_ini)
        conv_layer_4 = lasagne.layers.Conv2DLayer(conv_layer_3, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_4', W=W_ini)

        # pool
        pool_layer_2 = lasagne.layers.MaxPool2DLayer(conv_layer_4, pool_size=(2, 2), name='pool_2')

        # Dense layers
        dense_layer_1 = lasagne.layers.DenseLayer(pool_layer_2, num_units=128, W=W_ini,
                                                  name='dense_1', nonlinearity=lasagne.nonlinearities.rectify)
        dense_layer_1_dropout = lasagne.layers.DropoutLayer(dense_layer_1, p=self.dropout_frac, name='dense_1_dropout')
        dense_layer_2 = lasagne.layers.DenseLayer(dense_layer_1_dropout, num_units=128, W=W_ini,
                                                  name='dense_2', nonlinearity=lasagne.nonlinearities.rectify)
        dense_layer_2_dropout = lasagne.layers.DropoutLayer(dense_layer_2, p=self.dropout_frac, name='dense_2_dropout')

        # Initialize affine transform to identity
        b = np.zeros((2, 3), dtype=theano.config.floatX)
        b[0, 0] = 1
        b[1, 1] = 1

        # Final affine layer
        affine_layer = lasagne.layers.DenseLayer(dense_layer_2_dropout, num_units=6, W=lasagne.init.Constant(0.0),
                                                 b=b.flatten(), nonlinearity=lasagne.nonlinearities.identity,
                                                 name='affine')

        # Finally, create the transformer network
        transformer = lasagne.layers.TransformerLayer(incoming=input_layer,
                                                      localization_network=affine_layer,
                                                      downsample_factor=1)

        # Slice out the first channel
        transformer = lasagne.layers.SliceLayer(transformer, indices=0, axis=1)

        # Return
        self.transformer_graph = transformer


class stn_tps(generic_stn):
    """
    Class to instantiate a spatial transformer network with a thin plate
    spline transform
    """

    def initialize_cost(self):
        """
        Initializes the cost function
        """

        # get the transformed images
        predictions = lasagne.layers.get_output(self.transformer_graph, self.input_batch)

        # add in the cost (mse)
        self.cost = lasagne.objectives.squared_error(predictions, self.ref_imgs).mean()

        #### add in the cost (pixel weighted mse)
        # squared_error = lasagne.objectives.squared_error(predictions, self.ref_imgs)  # get squared error
        #
        # # Manipulate each ref image to be sum to 1
        # ref_shape = self.ref_imgs.shape  # Get shape for convenience
        # ref_img_reshape = T.reshape(self.ref_imgs, newshape=(ref_shape[0], ref_shape[1]*ref_shape[2]),
        #                             ndim=2)  # reshape to batch_size x num_pixels
        # # self.get_shape_1 = theano.function([self.ref_imgs], ref_img_reshape.shape)
        # ref_img_max = ref_img_reshape.max(axis=1).dimshuffle((0, 'x'))  # Get max for each image and create
        #                                                                 # broadcastable dimension
        # ref_img_min = ref_img_reshape.min(axis=1).dimshuffle((0, 'x'))  # get min for each image
        # # self.get_max_min_shape = theano.function([self.ref_imgs], [ref_img_max.shape, ref_img_min.shape])
        # ref_img_norm = (ref_img_reshape - ref_img_min) / (ref_img_max - ref_img_min)  # norm each image between 0 and 1
        # # self.get_norm_shape = theano.function([self.ref_imgs], ref_img_norm.shape)
        #
        # # self.get_ref_norm = theano.function([self.ref_imgs], ref_img_norm)
        # #
        # # self.get_norm_max_min = theano.function([self.ref_imgs], [ref_img_norm.max(axis=1), ref_img_norm.min(axis=1)])
        #
        # ref_img_partition = ref_img_norm / T.sum(ref_img_norm, axis=1).dimshuffle((0, 'x'))  # Partition so it sums to 1
        # # self.get_shape_2 = theano.function([self.ref_imgs], ref_img_partition.shape)
        # # norm_sum = T.sum(ref_img_norm, axis=1)
        # # part_sum = T.sum(ref_img_partition, axis=1)
        # # self.new_sum = theano.function([self.ref_imgs], [norm_sum, part_sum])
        # ref_img_partition = T.reshape(ref_img_partition, self.ref_imgs.shape)  # return to original shape
        #
        # # Aggregate using the normalized ref image as the weights
        # self.cost = 100000*(squared_error * ref_img_partition).mean()

        # create function
        self.get_cost = theano.function([self.input_batch, self.ref_imgs], self.cost)

    def initialize_process(self):
        """
        Initializes the process function
        """

        # Create symbolic output
        output = lasagne.layers.get_output(self.transformer_graph, self.input_batch, deterministic=True)

        # Create theano function
        self.process = theano.function([self.input_batch], output)

    def create_simple_network_graph(self, batch_size=32):
        """
        Builds a spatial transformer network
        :param batch_size: batch_size for optimization
        :return:
        """

        num_control_points = 16

        # Set initialization scheme
        if self.initialization == 'glorot_uniform':
            W_ini = lasagne.init.GlorotUniform('relu')
        elif self.initialization == 'glorot':
            W_ini = lasagne.init.Glorot('relu')
        elif self.initialization == 'glorot_normal':
            W_ini = lasagne.init.GlorotNormal('relu')
        elif self.initialization == 'he':
            W_ini = lasagne.init.He('relu')
        elif self.initialization == 'he_normal':
            W_ini = lasagne.init.HeNormal('relu')
        elif self.initialization == 'he_uniform':
            W_ini = lasagne.init.HeUniform('relu')
        elif self.initialization == 'orthogonal':
            W_ini = lasagne.init.Orthogonal('relu')
        else:
            raise AttributeError('Initialization string {} not understood'.format(self.initialization))

        # Input layer with size (batch_size, num_channels, height, width).
        # In our case, each channel will represent the image to change and the reference image.
        input_layer = lasagne.layers.InputLayer((batch_size, 2, 512, 512))

        # convolutions
        conv_layer_1 = lasagne.layers.Conv2DLayer(input_layer, num_filters=16, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_1', W=W_ini)
        conv_layer_2 = lasagne.layers.Conv2DLayer(conv_layer_1, num_filters=16, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_2', W=W_ini)

        # pool
        pool_layer_1 = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(2, 2), name='pool_1')

        # convolutions
        conv_layer_3 = lasagne.layers.Conv2DLayer(pool_layer_1, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_3', W=W_ini)
        conv_layer_4 = lasagne.layers.Conv2DLayer(conv_layer_3, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_4', W=W_ini)

        # pool
        pool_layer_2 = lasagne.layers.MaxPool2DLayer(conv_layer_4, pool_size=(2, 2), name='pool_2')

        # Dense layers
        dense_layer_1 = lasagne.layers.DenseLayer(pool_layer_2, num_units=128, W=W_ini,
                                                  name='dense_1', nonlinearity=lasagne.nonlinearities.rectify)
        dense_layer_1_dropout = lasagne.layers.DropoutLayer(dense_layer_1, p=self.dropout_frac, name='dense_1_dropout')
        dense_layer_2 = lasagne.layers.DenseLayer(dense_layer_1_dropout, num_units=128, W=W_ini,
                                                  name='dense_2', nonlinearity=lasagne.nonlinearities.rectify)
        dense_layer_2_dropout = lasagne.layers.DropoutLayer(dense_layer_2, p=self.dropout_frac, name='dense_2_dropout')

        # Slice out the first channel
        slice_layer = lasagne.layers.SliceLayer(input_layer, indices=0, axis=1)
        reshape_layer = lasagne.layers.ReshapeLayer(slice_layer, ([0], 1, [1], [2]))

        # Final tps layer
        tps_layer = lasagne.layers.DenseLayer(dense_layer_2_dropout, num_units=2*num_control_points,
                                              W=lasagne.init.Constant(0.0),
                                              b=lasagne.init.Constant(0.0),
                                              nonlinearity=lasagne.nonlinearities.identity,
                                              name='tps')

        # Finally, create the transformer network
        transformer = lasagne.layers.TPSTransformerLayer(incoming=reshape_layer,
                                                         localization_network=tps_layer,
                                                         downsample_factor=1,
                                                         control_points=num_control_points)

        # Return
        self.transformer_graph = transformer
