import numpy as np

def solve_steepest_descent(matrix, b, x, threshold):
    # Initial Residual (r = b - Ax)
    r = b - np.dot(matrix, x)
    
    iterations = 1000 
    
    for k in range(iterations):
        # Calculate the Euclidean norm of r
        r_norm = np.linalg.norm(r)
        
        # Stopping condition: if residue is close to 0
        if r_norm < threshold:
            print(f"\nConverged at iteration {k}")
            break

        # Calculate Ar (Matrix * r) - we use this in the denominator and the update step
        Ar = np.dot(matrix, r)
        
        # Calculate Alpha (Step Size)
        # Formula: (r . r) / (r . Ar)
        numerator = np.dot(r, r)
        denominator = np.dot(r, Ar)
        
        # Avoid division by zero
        if denominator == 0:
            break
            
        alpha = numerator / denominator
        
        # x(k) = x(k-1) + alpha * r(k-1)
        x = x + alpha * r
        
        # r(k) = r(k-1) - alpha * A * r(k-1)
        # We calculated 'A * r' earlier as 'Ar'
        r = r - alpha * Ar

        # Print status every iteration
        print(f"{k:<5} | {np.round(x, 4)} | {r_norm:.6f}")

    print("Final Solution x:", x)
    
    print(f"A * x = {np.dot(matrix, x)}")

def solve_pcgm(matrix, b, x, threshold):
    diagonal_elements = np.diag(matrix)
    P_inv = 1.0 / diagonal_elements 
    
    # r(0) = b - Ax
    r = b - np.dot(matrix, x)
    
    # z(0) = P^(-1) * r(0)
    z = P_inv * r 

    rho_current = np.dot(z, r)
    
    # Initialize p (will be set in the first iteration)
    p = np.zeros_like(b)
    
    # Parameters
    iterations = 1000

    for k in range(1, iterations + 1):
        # Check convergence (norm of r)
        r_norm = np.linalg.norm(r)
        if r_norm < threshold:
            print(f"\nConverged at iteration {k-1}")
            break

        if k == 1:
            # p(1) = z(0)
            p = z.copy()
        else:
            # beta(k) = (z(k-1) . r(k-1)) / (z(k-2) . r(k-2))
            # (z(k-2) . r(k-2)) as rho_prev
            beta = rho_current / rho_prev
            
            # p(k) = z(k-1) + beta * p(k-1)
            p = z + beta * p
        
        # Matrix * p(k)
        Ap = np.dot(matrix, p)
        
        #  p(k) . (A * p(k))
        denominator = np.dot(p, Ap)
        
        # Alpha = (z(k-1) . r(k-1)) / (p . Ap)
        alpha = rho_current / denominator
        
        # x(k) = x(k-1) + alpha * p(k)
        x = x + alpha * p
        
        # r(k) = r(k-1) - alpha * A * p(k)
        r = r - alpha * Ap
        
        # Save old rho for the next Beta calculation
        rho_prev = rho_current
        
        # z(k) = P^(-1) * r(k)
        z = P_inv * r
        
        # Update rho_current for the next loop (z(k) . r(k))
        rho_current = np.dot(z, r)

        print(f"{k:<5} | {np.round(x, 4)} | {r_norm:.6f}")

    print("-" * 50)
    print("Final Solution x:", x)
    
    # Calculate A * final_x to compare with b
    computed_b = np.dot(matrix, x)
    print("Verification (A * x):", computed_b)

def solve_cgm(matrix, b, x, threshold):
    k = 0
    
    # r = b - A * x
    r = b - np.dot(matrix, x)
    
    # spectral(new) = r(transpose) * r
    spectral_new = np.dot(r, r)
    
    # Calculate norm of b for the stopping condition
    norm_b = np.linalg.norm(b)

    # while radical(spectral) > threshold * euclidian norm (b)
    while np.sqrt(spectral_new) > threshold * norm_b:
        k += 1
        
        # Calculate Direction p
        if k == 1:
            p = r.copy()
        else:
            # beta = spectral(new) / spectral(old)
            beta = spectral_new / spectral_old
            
            # p = r + beta * p
            p = r + beta * p
        
        # w = A * p
        w = np.dot(matrix, p)
        
        # alpha = spectral(new) / (p(transpose) * w)
        # Note: dot product of p and w
        denominator = np.dot(p, w)
        alpha = spectral_new / denominator
        
        # x = x + alpha * p
        x = x + alpha * p
        
        # r = r - alpha * w
        r = r - alpha * w
        
        # spectral(old) = spectral(new)
        spectral_old = spectral_new
        
        # spectral(new) = r(transpose) * r
        spectral_new = np.dot(r, r)
        
        print(f"{k:<5} | {np.round(x, 4)} | {np.sqrt(spectral_new):.6f}")

    # --- Verification ---
    print("-" * 50)
    print("Final Solution x:", x)
    
    computed_b = np.dot(matrix, x)
    print("Computed b:      ", computed_b)

def solve_cgnr(matrix, b, x, threshold):
    # CGNR solves (A^T * A) x = A^T * b
    # Ideal for non-symmetric matrices
    
    # r = b - A * x
    r = b - np.dot(matrix, x)
    
    # z = A^T * r
    z = np.dot(matrix.T, r)
    
    # p = z
    p = z.copy()
    
    iterations = 1000
    norm_b = np.linalg.norm(b)

    print("\n--- Starting CGNR ---")

    for k in range(iterations):
        r_norm = np.linalg.norm(r)
        
        # Stopping condition
        if r_norm < threshold: # Or (r_norm / norm_b) < threshold
            print(f"Converged at iteration {k}")
            break
        
        # w = A * p
        w = np.dot(matrix, p)
        
        # Alpha calc: (z . z) / (w . w)
        z_dot_z = np.dot(z, z)
        w_dot_w = np.dot(w, w)
        
        if w_dot_w == 0: break
            
        alpha = z_dot_z / w_dot_w
        
        # Update x: x = x + alpha * p
        x = x + alpha * p
        
        # Update r: r = r - alpha * w
        r = r - alpha * w
        
        # Update z_new: z_new = A^T * r_new
        z_new = np.dot(matrix.T, r)
        
        # Beta calc: (z_new . z_new) / (z . z)
        z_new_dot = np.dot(z_new, z_new)
        beta = z_new_dot / z_dot_z
        
        # Update p: p = z_new + beta * p
        p = z_new + beta * p
        
        # Update z for next iteration
        z = z_new
        
        print(f"{k:<5} | {np.round(x, 4)} | {r_norm:.6f}")

    print("Final Solution x (CGNR):", x)


def solve_cgne(matrix, b, x, threshold):
    # CGNE solves A * A^T * y = b, where x = A^T * y
    # Minimizes the error norm ||x - x*||
    
    # r = b - A * x
    r = b - np.dot(matrix, x)
    
    # p = A^T * r
    p = np.dot(matrix.T, r)
    
    iterations = 1000

    print("\n--- Starting CGNE ---")

    for k in range(iterations):
        r_norm = np.linalg.norm(r)
        
        if r_norm < threshold:
            print(f"Converged at iteration {k}")
            break
            
        # Alpha calc: (r . r) / (p . p)
        r_dot_r = np.dot(r, r)
        p_dot_p = np.dot(p, p)
        
        if p_dot_p == 0: break
            
        alpha = r_dot_r / p_dot_p
        
        # Update x: x = x + alpha * p
        x = x + alpha * p
        
        # Update r: r = r - alpha * A * p
        Ap = np.dot(matrix, p)
        r = r - alpha * Ap
        
        # Beta calc: (r_new . r_new) / (r_old . r_old)
        r_new_dot = np.dot(r, r)
        beta = r_new_dot / r_dot_r
        
        # Update p: p = A^T * r + beta * p
        # Note: We need A^T * r_new here
        At_r = np.dot(matrix.T, r)
        p = At_r + beta * p
        
        print(f"{k:<5} | {np.round(x, 4)} | {r_norm:.6f}")

    print("Final Solution x (CGNE):", x)

def run_ex4():
    print("-" * 50)
    print("Exercise 4: CGNR")
    
    matrix = np.array([
        [60.0, 10.0, 5.0, 2.0, 1.0],
        [10.0, 55.0, 4.0, 2.0, 1.0],
        [5.0, 4.0, 50.0, 6.0, 2.0],
        [2.0, 2.0, 6.0, 45.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 40.0]
    ])
    b = np.array([264.0, 289.0, 331.0, 335.0, 315.0])
    x = np.zeros(5)
    threshold = 1e-10
    
    solve_cgnr(matrix, b, x, threshold)

def run_ex5():
    print("-" * 50)
    print("Exercise 5: CGNE")
    
    matrix = np.array([
        [60.0, 10.0, 5.0, 2.0, 1.0],
        [10.0, 55.0, 4.0, 2.0, 1.0],
        [5.0, 4.0, 50.0, 6.0, 2.0],
        [2.0, 2.0, 6.0, 45.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 40.0]
    ])
    b = np.array([264.0, 289.0, 331.0, 335.0, 315.0])
    x = np.zeros(5)
    threshold = 1e-10
    
    solve_cgne(matrix, b, x, threshold)

def run():
    matrix = np.array([
        [60.0, 10.0, 5.0, 2.0, 1.0],
        [10.0, 55.0, 4.0, 2.0, 1.0],
        [5.0, 4.0, 50.0, 6.0, 2.0],
        [2.0, 2.0, 6.0, 45.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 40.0]
    ])
    
    b = np.array([264.0, 289.0, 331.0, 335.0, 315.0],)
    
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    threshold = 10e-10
    # solve_steepest_descent(matrix, b, x, threshold)
    # solve_pcgm(matrix, b, x, threshold)
    solve_cgm(matrix, b, x, threshold)

if __name__ == "__main__":
    run()
    run_ex4()
    run_ex5()