import numpy as np

def iteration_step(A: np.ndarray, b: np.ndarray, x: np.ndarray, r: np.ndarray, w: int, P: np.ndarray):
    z = get_iteration_corector(P, r)

    # Update the approximation
    x_new = get_iteration_aprox(x, z, w)

    # Update the residual
    r_new = get_iteration_residue(A, w, r, z)

    return x_new, r_new, z


def get_iteration_aprox(x: np.ndarray, z: np.ndarray, w: int):
    new_x = x + w * z
    return new_x


def get_iteration_residue(A: np.ndarray, w: int, old_r: np.ndarray, z: np.ndarray):
    new_r = old_r - w * (A @ z)
    return new_r


def get_initial_residue(A: np.ndarray, b: np.ndarray, x: np.ndarray):
    r0 = b - A @ x
    return r0

def get_iteration_corector(P: np.ndarray, r: np.ndarray):
    z = np.linalg.solve(P, r)
    return z

def euclidean_norm(v: np.ndarray):
    return np.linalg.norm(v)

def run():
    # Size of the system
    n = 3  # You can change this as needed

   # Matrix A (3x3)
    A = np.array([
         [3, 0,  4],
         [7, 4, 2],
         [-1, 1, 2
          ]
      ], dtype=float)

   # Vector b (3x1)
    b = np.array([
      [6],
      [-7],
      [-14]
   ], dtype=float)

    # Initial approximation x = 0 (n x 1)
    x = np.zeros((n, 1))

    # Initial residual r0 = b - A @ x
    r = get_initial_residue(A, b, x)

    # Preconditioner P
    P_Jacobi = np.diag(np.diag(A))
    P_GSA = np.tril(A)
    P_GSB = np.triu(A)

    # Iteration parameter
    w = 1  # You can adjust as needed

    # Max step
    max_no_steps = 1000

    tolerance = 1e-14

    # Perform one iteration step
    step = 0
    while step < max_no_steps:
      step += 1
      new_x, new_r, new_z = iteration_step(A, b, x, r, w, P_Jacobi)
      if euclidean_norm(new_r) < tolerance:
          print("Early stoppage: ", step)
          break
      x, r, z = new_x, new_r, new_z
    print("Result: ", x)


if __name__ == '__main__':
   run()    
    