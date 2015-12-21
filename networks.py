import numpy as np
import theano
import lasagne
import theano.tensor as T


class stn:
    """
    Class to instantiate a spatial transformer network
    """

    def __init__(self, batch_size=32, alpha=0.001, max_norm=5):

        # Initialize parameters
        self.batch_size = batch_size
        self.alpha = alpha
        self.max_norm = max_norm

        # Create the graph
        self.create_network_graph(batch_size=self.batch_size)

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

        # Create adam function
        updates = lasagne.updates.adam(updates,
                                       self.params_to_train,
                                       learning_rate=self.alpha)

        # Create train function
        self.train_adam = theano.function([self.input_batch, self.ref_imgs],
                                          self.cost,
                                          updates=updates)

    def create_inputs(self):
        """
        Initializes the tensors for the input images and reference images
        """
        # Create input tensor
        self.input_batch = T.tensor4('input_batch', dtype=theano.config.floatX)

        # Create reference tensor (batch_size, 1, height, width)
        self.ref_imgs = T.tensor3('ref_imgs', dtype=theano.config.floatX)

    def initialize_cost(self):
        """
        Initializes the cost function
        """

        # get the transformed images
        predictions = lasagne.layers.get_output(self.transformer_graph, self.input_batch)

        # add in the cost
        self.cost = lasagne.objectives.squared_error(predictions, self.ref_imgs).mean()

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

    def create_network_graph(self, batch_size=32):
        """
        Builds a spatial transformer network
        :param batch_size:
        :return:
        """

        # Set initialization scheme
        W_ini = lasagne.init.GlorotUniform()

        # Input layer with size (batch_size, num_channels, height, width).
        # In our case, each channel will represent the image to change and the reference image.
        input_layer = lasagne.layers.InputLayer((batch_size, 2, 512, 512))

        # convolutions
        conv_layer_1 = lasagne.layers.Conv2DLayer(input_layer, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_1', W=W_ini)
        conv_layer_2 = lasagne.layers.Conv2DLayer(conv_layer_1, num_filters=32, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_2', W=W_ini)

        # pool
        pool_layer_1 = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(2, 2), name='pool_1')

        # more convolutions
        conv_layer_3 = lasagne.layers.Conv2DLayer(pool_layer_1, num_filters=64, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_3', W=W_ini)
        conv_layer_4 = lasagne.layers.Conv2DLayer(conv_layer_3, num_filters=64, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_4', W=W_ini)

        # pool
        pool_layer_2 = lasagne.layers.MaxPool2DLayer(conv_layer_4, pool_size=(2, 2), name='pool_2')

        # more convolutions
        conv_layer_5 = lasagne.layers.Conv2DLayer(pool_layer_2, num_filters=128, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_5', W=W_ini)
        conv_layer_6 = lasagne.layers.Conv2DLayer(conv_layer_5, num_filters=128, filter_size=(3, 3),
                                                  stride=1, pad='full', name='conv_6', W=W_ini)

        # pool
        pool_layer_3 = lasagne.layers.MaxPool2DLayer(conv_layer_6, pool_size=(2, 2), name='pool_3')

        # Dense layers
        dense_layer_1 = lasagne.layers.DenseLayer(pool_layer_3, num_units=128, W=W_ini,
                                                  name='dense_1')

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
