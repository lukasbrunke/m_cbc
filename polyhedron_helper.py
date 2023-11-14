import numpy as np
import matplotlib.pyplot as plt


def is_inside_polyhedron(x, X):
    """
    Checks if a point is inside a polyhedron.

    :param x: The point to be checked.
    :param X: The polyhedron.
    :return: True if the point is inside the polyhedron, False otherwise.
    """

    A_hull, b_hull = X

    is_inside = []
    for i in range(A_hull.shape[0]):
        is_inside.append(A_hull[i, :].T @ x <= - (b_hull[i] + 1e-5))

    if all(not bool_ for bool_ in is_inside):
        is_inside = True
    else:
        is_inside = False

    return is_inside


def polyhedron2vertices(X):
    """
    Calculates a polyhedron's vertices based on its halfspace representation.

    :param X: The polyhedron, X = (A, b).
    :return: The vertices of the polyhedron.
    """

    A, b = X

    # Get the number of dimensions
    n = A.shape[1]

    # Get the number of halfspaces
    m = A.shape[0]

    # Get the vertices of the polyhedron
    vertices_x = []
    vertices_y = []
    for i in range(m):
        for j in range(m):
            # Avoid duplicates
            if i < j:
                # Get the intersection of the two halfspaces
                A_i = A[i, :].reshape((1, n))
                A_j = A[j, :].reshape((1, n))
                A_ = np.vstack((A_i, A_j))
                b_ = np.array([b[i], b[j]])
                try:
                    vertex = np.linalg.solve(A_, b_)
                    vertices_x.append(vertex[0])
                    vertices_y.append(vertex[1])
                except:
                    # no solution exists
                    continue
                    
    return vertices_x, vertices_y


def plot_polyhedra2d(x_linspace, y_linspace, X, linestyle='k-'):
    """
    Plots the half space representation of a polyhedron in 2D.

    :param x_linspace: The x-axis values.
    :param y_linspace: The y-axis values.
    :param X: The polyhedron, X = (A, b).
    :param linestyle: The linestyle of the polyhedron.
    """

    A_hull, b_hull = X
    for i in range(A_hull.shape[0]):
        if abs(A_hull[i, 1]) < 1e-5:
            y = y_linspace
            x = (- b_hull[i] - A_hull[i, 1] * y) / A_hull[i, 0]
        else:
            x = x_linspace
            y = (- b_hull[i] - A_hull[i, 0] * x) / A_hull[i, 1]
        plt.plot(x, y, linestyle)


def plot_solid_polyhedra2d(X, color='k', alpha=1.0):
    """
    Plots a solid polyhedron in 2D.

    :param X: The polyhedron, X = (A, b).
    :param color: The color of the polyhedron.
    :param alpha: The transparency of the polyhedron.
    """

    vertices_x, vertices_y = polyhedron2vertices(X)
    vertices_x, vertices_y = np.array(vertices_x), np.array(vertices_y)

    # plot the polyhedra
    order = np.argsort(np.arctan2(vertices_y - vertices_y.mean(), 
                                  vertices_x - vertices_x.mean()))
    plt.fill(vertices_x[order], vertices_y[order], color, alpha=alpha)
