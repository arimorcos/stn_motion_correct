import os
import re
import tiff_helpers
from random import shuffle
import numpy as np
import theano


class image_manager:
    """
    Class to manage images for convolutional spatial transformer network
    """

    def __init__(self, folder_list=None, batch_size=16,
                 num_files_per_batch=4, num_batches_per_file_group=400,
                 image_dim=(512, 512)):
        """
        Initializes manager class
        :param folder_list: list of folders to source files from
        :param batch_size: batch_size of images to offer
        :param num_batches_per_file_group: Number of batches for each group of files before advancing
        :param num_files_per_batch: Number of files to include in each file group
        :param image_dim: Image dimensions
        :return:
        """
        # Initialize parameters
        self.batch_size = batch_size
        self.num_files_per_batch = num_files_per_batch
        self.num_batches_per_file_group = num_batches_per_file_group
        self.image_dim = image_dim
        self.curr_batch = 0
        self.curr_data = None
        self.curr_ref = None

        # Set folder list
        if folder_list is None:
            folder_list = self.get_defined_folder_list()

        # Get file list from folder list
        self.get_file_list(folder_list)

        # Create file schedule
        self.create_schedule()

    @staticmethod
    def get_defined_folder_list():
        """
        Place to pre-specify a folder-list (default)
        :return: returns the folder list
        """

        folder_list = ['/media/arimorcos/SSD 1']
        return folder_list

    def get_file_list(self, folder_list):
        """
        Recursively looks through folder list to get a list of all .tif files for training
        :param folder_list: list of root folders
        :return: List of matching .tif files
        """

        file_list = []

        # loop through each folder
        for folder in folder_list:
            for (file_dir, _, files) in os.walk(folder):
                for f in files:
                    path = os.path.join(file_dir, f)
                    file_list.append(path)
                    # print path

        # Remove files containing .Trash-100
        file_list = [file for file in file_list if not re.match(".*.Trash-1000.*", file)]
        file_list = [file for file in file_list if re.match(".*AM\d{3}_\d{3}_\d{3}.tif", file)]

        # Store
        self.file_list = file_list

    def set_batch_size(self, batch_size):
        """
        Sets the new batch size
        :param batch_size: new batch_size. Must be an integer.
        """

        # Check if batch size is integer
        if batch_size is not int:
            raise TypeError("Provided batch size must be of type int")

        # store new one
        self.batch_size = batch_size

    def get_batch_size(self):
        """
        Retrieves the batch size
        :return: Current batch size
        """
        return self.batch_size

    def create_schedule(self):
        """
        Creates a schedule for which images to load and use
        :return:
        """

        # Shuffle file list
        self.file_schedule = list(self.file_list)
        shuffle(self.file_schedule)
        self.curr_file = 0

    def advance_schedule(self):
        """
        Advances the schedule, checks if it needs to load new files, loads those files and sets the curr_batch to a
        new batch
        :return:
        """

        # Check if we need to load in data (If we've reached the limit for the current file group)
        if self.curr_batch % self.num_batches_per_file_group == 0:

            # Load in some new files
            load_list = self.file_schedule[self.curr_file:self.curr_file + self.num_files_per_batch]

            # Initialize tiff array
            tiff_data = []

            # Loop through and load
            for load_file in load_list:
                tiff_data.append(
                                tiff_helpers.read_tiff(load_file, np.arange(1, 2000, 2))
                                )

            # Store
            self.tiff_data = tiff_data

            # Increment curr file
            self.curr_file += self.num_files_per_batch

        # Initialize new batch
        new_data = np.empty(shape=(self.batch_size, 2, self.image_dim[0], self.image_dim[1]),
                            dtype=theano.config.floatX)

        # Create new batch
        for batch_index, image in enumerate(range(self.batch_size)):

            # Choose file randomly
            which_file = np.random.randint(0, high=self.num_files_per_batch)

            # Choose test and reference image
            image_indices = np.random.choice(1000, size=(2, 1), replace=False)

            # Crop out, permute to appropriate indices and add
            to_add = self.tiff_data[which_file][:, :, image_indices]
            to_add = tiff_helpers.mean_center_img(to_add)
            to_add = np.transpose(to_add, axes=(3, 2, 0, 1))
            new_data[batch_index, :, :, :] = to_add

        # Increment
        self.curr_batch += 1

        # Create current reference image
        self.curr_ref = new_data[:, 1, :, :]

        # Store
        self.curr_data = new_data

    def offer_data(self):
        """
        Returns the current batch
        :return: current batch
        """

        if self.curr_data is None:
            raise BaseException("Must advance schedule first")

        return self.curr_data, self.curr_ref
