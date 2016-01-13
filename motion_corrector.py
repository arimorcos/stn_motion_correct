import networks
import tiff_helpers
import numpy as np
from theano import config


class motion_corrector:
    """
    Handles the actual motion correction of tiff files. Automatically interfaces with a trained spatial transformer
    network and contains methods to motion correct individual files.
    """
    default_param_file = '/media/arimorcos/4TB External/stn_conv_net/160112_152116/Epoch_0400_weights.pkl'

    def __init__(self, batch_size=128, param_file=None):
        """
        Initialization
        :param batch_size: batch size for feedforward run of motion correction
        """

        # Parse param_file
        if param_file is None:
            param_file = self.default_param_file

        # Initialize network and set parameters
        self.initialize_network(batch_size)
        self.set_network_params(param_file)

        # Store relevant params
        self.__batch_size__ = batch_size

        pass

    @staticmethod
    def get_tiff(file_to_load):
        """
        Wrapper for load tiff
        :param file_to_load: Path to the file to load
        :return: Loaded tiff
        """
        # Get number of pages and load tiff (hardcoded currently to every other)
        num_pages = tiff_helpers.get_num_tiff_pages(file_to_load)
        tiff = tiff_helpers.read_tiff(file_to_load, pages=range(1, num_pages, 2)).astype(config.floatX)

        # mean center tiff
        tiff = tiff_helpers.mean_center_img(tiff)

        return tiff

    def correct_file(self, file_to_correct, ref_frame):
        """
        Method to apply motion correction to a given file
        :param file_to_correct: path to a file
        :param ref_frame: integer index of the frome to align to
        :return:
            corrected_tiff: a numpy array containing the motion corrected file
            tiff: the original tiff
        """

        tiff = self.get_tiff(file_to_correct)

        corrected_tiff = self.correct_tiff(tiff, ref_frame)

        return corrected_tiff, tiff

    def correct_tiff(self, tiff, ref_frame):
        """
        Applies motion correction to a given tiff
        :param tiff: tiff file (must be mean-centered)
        :param ref_frame: integer index of the frame to align to
        :return: corrected tiff file
        """
        # extract reference frame
        ref_img = tiff[:, :, ref_frame]

        # Get number of usable frames
        num_frames = tiff.shape[2]

        # Get number of batches
        num_batches = int(np.ceil(float(num_frames) / self.__batch_size__))

        # Convert ref image to appropriate shape
        ref_imgs = np.tile(np.expand_dims(ref_img, 2), (1, 1, self.__batch_size__))
        ref_imgs = np.transpose(np.expand_dims(ref_imgs, 3), axes=(2, 3, 0, 1))

        # Loop through each batch and correct
        for batch_ind in range(num_batches):

            # Get batch_frames
            start_ind = batch_ind*self.__batch_size__
            stop_ind = (batch_ind + 1)*self.__batch_size__
            batch_frames = tiff[:, :, start_ind:stop_ind]

            # Check if last batch
            if batch_frames.shape[2] < self.__batch_size__:
                # batch_frames[:, :, batch_frames.shape[2]:self.__batch_size__] = 0
                batch_frames = np.pad(batch_frames, ((0, 0), (0, 0), (0, self.__batch_size__ - batch_frames.shape[2])),
                                      mode='constant', constant_values=0)

            # Convert to appropriate indices (batch_size, num_channels, height, width)
            to_correct = np.expand_dims(batch_frames, 3)
            to_correct = np.transpose(to_correct, axes=(2, 3, 0, 1))
            to_correct = np.concatenate((to_correct, ref_imgs), 1)

            # Run through the network
            if batch_ind == 0:
                corrected_tiff = self.stn.process(to_correct)
            else:
                temp_corrected = self.stn.process(to_correct)
                corrected_tiff = np.concatenate((corrected_tiff, temp_corrected), axis=0)

        # Return to height x width x pages
        corrected_tiff = np.transpose(corrected_tiff, axes=(1, 2, 0))

        # Crop to appropriate size
        corrected_tiff = corrected_tiff[:, :, 0:num_frames]

        return corrected_tiff

    def initialize_network(self, batch_size):
        """
        Creates a network
        :return: network object
        """

        self.stn = networks.stn(batch_size=batch_size)

    def set_network_params(self, param_file):
        """
        Sets the netowrk parameters to that contained in a given file
        :param param_file: path to the parameter file
        """

        self.stn.set_parameters(param_file=param_file)



