import numpy as np
from tifffile import TiffFile
import warnings
import matplotlib.pyplot as plt


def read_tiff(path, pages=None):
    """
    Reads in a tiff stack
    :param path: Full path to file
    :param pages: list or numpy array of pages to load
    :return: height x width x num_pages array of image files
    """

    # Get number of requested pages
    if pages is None:
        num_pages = 1
    elif type(pages) is int:
        num_pages = 1
        pages = [pages]
    elif type(pages) is list or type(pages) is np.ndarray:
        num_pages = len(pages)
    else:
        raise TypeError('Pages is type {}, but must be a list, int, or array'.format(type(pages).__name__))

    with TiffFile(path) as f:

        # get number of pages in actual tiff
        num_tiff_pages = len(f.pages)
        if num_pages > num_tiff_pages:
            raise IndexError("Too many pages requested. Requested {} pages but only {} pages in tiff"
                             .format(num_pages, num_tiff_pages))
        if pages is None and num_tiff_pages > 1:
            warnings.warn("No specific pages requested, so returning all pages ({})"
                          .format(num_tiff_pages))
            pages = xrange(num_tiff_pages)
            num_pages = num_tiff_pages

        # initialize tiff array
        tiff_shape = f.pages[0].shape
        tiff_array = np.empty(shape=(tiff_shape[0], tiff_shape[1], num_pages))

        # load each page and store
        for ind, page in enumerate(pages):
            curr_page = f.pages[page]
            tiff_array[:, :, ind] = curr_page.asarray()

    # Compress if only 2d
    if tiff_array.shape[2] == 1:
        tiff_array = tiff_array.squeeze(axis=2)

    return tiff_array


def get_num_tiff_pages(path):
    """
    Returns the number of pages in the tiff
    :param path: path to the tiff to query
    :return int: number of pages in tiff stack
    """

    with TiffFile(path) as f:
        return len(f.pages)


def imshowpair(img_1, img_2, ax=None):
    """
    Shows images as an overlapping pair with image 1 in red and image 2 in green
    :param img_1: first image to display. Will be shown in red.
    :param img_2: second image to display. Will be shown in green.
    :param ax: axis to plot on. If none, current.
    :return: None
    """

    if ax is None:
        ax = plt.gca()

    # Create zero image
    zero_pad = np.zeros(shape=img_1.shape)

    # Normalize images
    img_1 = norm_img(img_1)
    img_2 = norm_img(img_2)

    # Concatenate image
    show_img = np.stack((img_1, img_2, zero_pad), axis=2)

    # Plot
    ax.imshow(show_img)


def norm_img(img):
    """
    Normalizes an image to between 0 and 1
    :param img: image to normalize
    :return: nomralized image
    """

    return (img - img.min())/(img.max() - img.min())


def mean_center_img(img):
    """
    Makes each image in a stack have zero mean and unit std
    :param img: 3d array of images
    :return: normalized image array
    """

    for page in range(img.shape[2]):
        temp_page = img[:, :, page]
        img[:, :, page] = (temp_page - temp_page.mean())/temp_page.std()

    return img


