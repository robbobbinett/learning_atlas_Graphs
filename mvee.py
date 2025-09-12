import cvxpy as cp
import numpy as np


def mvee(X):
    """
    Compute the minimum-volume enclosing ellipsoid for a set of points.
    Ellipsoid takes the following form: x.T @ A x + c = 1

    Input:
    X (np.ndarray): array of shape (n, d), where n is the number of points
                    and d is the dimensionality of the points

    Output:
    A (np.ndarray): array of shape (d, d)
    b (np.ndarray): array of shape (d,)
    """
    n, d = X.shape

    # Define semidefinite cone problem
    P = cp.Variable((d, d))
    c = cp.Variable(d)
    Z = cp.Variable((d, d))
    v = cp.Variable(d)

    objective = cp.Maximize(cp.sum(v))

    ### Constraints
    constraints = []
    ##### v <= log(diag(Z))
    for j in range(d):
        constraints.append(v[j] <= cp.log(Z[j, j]))
    ##### Block matrix constraint
    upper_block = cp.hstack([P, Z])
    lower_block = cp.hstack([Z.T, cp.diag(cp.diag(Z))])
    block_matrix = cp.vstack([upper_block, lower_block])
    constraints.append(block_matrix >> 0)
    ##### Ellipsoid constraints: ||P @ X[j] - c||_2 <= 1
    for j in range(n):
        constraints.append(cp.norm(P @ X[j] - c, 2) <= 1)
#        constraints.append(cp.quad_form(X[j]- c, P) <= 1)
    ##### PD constraint
    constraints.append(P >> 0)
    ##### Z lower-triangular
    for j in range(d):
        for k in range(j+1, d):
            constraints.append(Z[j, k] == 0)

    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return solution
    Pinv = np.linalg.inv(P.value)
    A = Pinv.T @ Pinv
    b = 2 * A @ c.value

    return A, b
