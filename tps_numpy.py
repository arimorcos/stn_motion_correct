import numpy as np
from tps import from_control_points


def U_func(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)


def get_transformed_point(new_x, new_y, source_points, coefficients):
    """
    Calculates the transformed point's value using the provided coefficients
    :param new_x: x point to transform
    :param new_y: y point to transform
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (2, control_points + 3))
    :return:
    """

    # Calculate the affine portion
    values = np.zeros(2)
    for var in range(2):
        values[var] = coefficients[var, 0] + coefficients[var, 1]*new_x + coefficients[var, 2]*new_y

    # Add in the bending energy portion
    num_points = source_points.shape[1]
    for point in range(num_points):
        temp_distance = U_func(new_x, new_y, source_points[0, point], source_points[1, point])
        for var in range(2):
            values[var] += coefficients[var, point + 3] * temp_distance

    return values


if __name__ == "__main__":

    # Create source grid
    num_control_points = 16
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(np.linspace(-1, 1, grid_size),
                                                     np.linspace(-1, 1, grid_size))

    # Create destination grid
    x_offset = 0.5
    y_offset = 0.1
    x_control_dest = x_control_source.flatten() + x_offset
    y_control_dest = y_control_source.flatten() + y_offset

    """
    Add Points
        We will have the number of variables, in this case 2 (x and y), hardcoded
    """

    # Create 2 x n array of x and y source variables
    source_points = np.vstack((x_control_source.flatten(), y_control_source.flatten()))

    # Create 2 x n array of x and y destination variables
    dest_points = np.vstack((x_control_dest, y_control_dest))

    """
    Solve the equation
    """

    # Get number of equations
    num_equations = num_control_points + 3

    # Initialize L to be num_equations square matrix
    L = np.zeros((num_equations, num_equations))

    # Loop through and create P matrix components
    for point in range(num_control_points):

        # Set each of the first three rows to the x and y points (here, we're creating the P matrix part)
        # L[0, point + 3] = 1.
        # L[1, point + 3] = source_points[0, point]
        # L[2, point + 3] = source_points[1, point]
        L[0, point + 3] = 1.
        L[1, point + 3] = source_points[0, point]
        L[2, point + 3] = source_points[1, point]

        # Set each of the first three columns to the x and y points (this is P transpose)
        L[point + 3, 0] = 1.
        L[point + 3, 1] = source_points[0, point]
        L[point + 3, 2] = source_points[1, point]

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = U_func(source_points[0, point_1], source_points[1, point_1],
                                                 source_points[0, point_2], source_points[1, point_2])

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Now that we have the L matrix, let's actually calculate things

    # First, invert the matrix
    L_inv = np.linalg.inv(L)

    # Calculate the coefficients for each variable (a_1, a_x, a_y)
    coefficients = np.zeros((2, num_equations))
    for variable in range(2):
        for eq_1 in range(num_equations):
            for eq_2 in range(num_equations):
                coefficients[variable, eq_1] += L_inv[eq_1, eq_2] * dest_points[variable, eq_2 - 3]

    # Create test points
    test_points = np.random.normal(0, 2, size=(10, 2))

    # Transform points
    np_transformed_points = np.empty(test_points.shape)
    for ind, (x, y) in enumerate(test_points):
        np_transformed_points[ind, :] = get_transformed_point(x, y, source_points, coefficients)

    """
    Get C code version
    """
    full_array = np.hstack((source_points.T, dest_points.T))
    full_array = full_array.tolist()

    # Create transformer
    tps_obj = from_control_points(full_array)

    # Transform
    c_transformed_points = np.empty(test_points.shape)
    for ind, (x, y) in enumerate(test_points):
        c_transformed_points[ind, :] = tps_obj.transform(x, y)

    """
    Check if equal
    """
    print np.allclose(np_transformed_points, c_transformed_points)


