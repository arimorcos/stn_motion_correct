import numpy as np
import scipy.interpolate as interpolate
from tps import from_control_points
from tiff_helpers import norm_img


def apply_tps_transform(img, dest_control_points):
    """
    Applies the thin-plate spline transform to the image provided and outputs the transformed image
    :param img: The image to be transformed
    :param dest_control_points: num_control_points x 2 array of x and y coordinates for control points
    :return: transformed image with the same dimensions as the input image
    """

    # get the image dimensions
    height, width = img.shape

    # normalize
    img = norm_img(img)

    # Get number of control points
    num_control_points = dest_control_points.shape[0]

    # Generate the start control points
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(np.linspace(-1, 1, grid_size),
                                                     np.linspace(-1, 1, grid_size))
    source_control_points = np.vstack((x_control_source.flatten(), y_control_source.flatten())).T

    # Create control matrix comprising a list of 4-element lists, with the following elements:
    # [source_x, source_y, dest_x, dest_y]
    full_array = np.hstack((source_control_points, dest_control_points))
    full_array = full_array.tolist()

    # Create the transformer object
    tps_transformer = from_control_points(full_array)

    # Generate the input grid
    x_source, y_source = np.meshgrid(np.linspace(-1, 1, height),
                                     np.linspace(-1, 1, width))
    x_source = x_source.flatten()
    y_source = y_source.flatten()

    # transform each point pair
    x_dest = np.empty(x_source.shape)
    y_dest = np.empty(y_source.shape)
    for ind, (x_coord, y_coord) in enumerate(zip(x_source, y_source)):
        x_dest[ind], y_dest[ind] = tps_transformer.transform(x_coord, y_coord)

    # Convert destination points to appropriate width and height
    x_dest_norm = (x_dest + 1) / 2
    y_dest_norm = (y_dest + 1) / 2
    x_dest_norm *= height
    y_dest_norm *= width

    # Get values at floored points
    # round_vals = np.zeros(shape=(height, width)).ravel()
    # x_dest_floor = np.floor(x_dest_norm).astype('int32')
    # y_dest_floor = np.floor(y_dest_norm).astype('int32')
    # use_ind = (x_dest_floor >= 0) & (x_dest_floor <= height - 1) & (y_dest_floor >= 0) & (y_dest_floor <= width - 1)
    # x_dest_round_use = x_dest_floor[use_ind]
    # y_dest_round_use = y_dest_floor[use_ind]
    # ravel_use_ind = np.ravel_multi_index((x_dest_round_use, y_dest_round_use), (height, width))
    # round_vals[ravel_use_ind] = img.ravel()[ravel_use_ind]
    # round_vals = np.reshape(round_vals, (height, width))

    # Interpolate
    # interpolator = interpolate.RectBivariateSpline(range(height), range(width), round_vals)
    # interpolator = interpolate.interp2d(range(height), range(width), round_vals, fill_value=0)
    # interp_image = interpolator.ev(x_dest_norm, y_dest_norm).reshape((height, width)).T
    # interp_image = interpolator(range(height), range(width)).reshape((height, width))
    # round_vals = []
    interpolator = interpolate.RectBivariateSpline(range(height), range(width), img)
    interp_image = interpolator.ev(x_dest_norm, y_dest_norm).reshape((height, width)).T

    # Zero bad values
    # bbox = np.array([np.unravel_index(np.argmax(x_dest_norm >= 0), (height, width))[1],
    #                  np.unravel_index(np.argmin(x_dest_norm <= height - 1), (height, width))[1],
    #                  np.unravel_index(np.argmax(y_dest_norm >= 0), (height, width))[0],
    #                  np.unravel_index(np.argmin(y_dest_norm >= 0), (height, width))[0]])
    # if bbox[1] == 0:
    #     bbox[1] = height
    # if bbox[3] == 0:
    #     bbox[3] = width
    # interp_image[:bbox[0], :] = 0
    # interp_image[bbox[1]:, :] = 0
    # interp_image[:, :bbox[2]] = 0
    # interp_image[:, bbox[3]:] = 0

    # return x_source, y_source, x_dest_norm, y_dest_norm, round_vals, interp_image
    return interp_image