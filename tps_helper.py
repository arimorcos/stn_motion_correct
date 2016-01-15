import numpy as np
from tps import from_control_points


def apply_tps_transform(self, img, control_points):
    """
    Applies the thin-plate spline transform to the image provided and outputs the transformed image
    :param img: The image to be transformed
    :param control_points: num_control_points x 2 array of x and y coordinates for control points
    :return: transformed image with the same dimensions as the input image
    """

    # get the image dimensions
    height, width = img.shape

    # Get number of control points
    num_control_points = control_points.shape[0]

    # Generate the start control points
    grid_size = np.sqrt(num_control_points)
    x_control_t, y_control_t = np.meshgrid(np.linspace(-1, 1, grid_size),
                                           np.linspace(-1, 1, grid_size))

    # Create control matrix comprising a list of 4-element lists, with the following elements:
    # [source_x, source_y, dest_x, dest_y]
    control_points = np.reshape(control_points, (-1, 2))
    full_array = np.vstack((x_control_t.flatten(), y_control_t.flatten(), control_points)).tolist()

    # Create the transformer object
    tps_transformer = from_control_points(full_array)

    # Generate the input grid
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, height),
                           np.linspace(-1, 1, width))
    x_t = x_t.flatten()
    y_t = y_t.flatten()

    # transform each point pair
    x_s = np.empty(x_t.shape)
    y_s = np.empty(y_t.shape)
    for ind, (x_coord, y_coord) in enumerate(zip(x_t, y_t)):
        x_s[ind], y_s[ind] = tps_transformer.transform(x_coord, y_coord)