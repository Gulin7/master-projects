import numpy as np

def solve_steepest_descent(matrix, b, x0, itmax, TOL):
    matrix = np.array(matrix, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    # Initial Residual (r = b - Ax)
    r = b - np.dot(matrix, x)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0

    for k in range(itmax):
        # Calculate the Euclidean norm of r
        r_norm = np.linalg.norm(r)

        # Stopping condition: normalized residue
        if r_norm / norm_b < TOL:
            return x, k

        # Calculate Ar (Matrix * r)
        Ar = np.dot(matrix, r)

        # Calculate Alpha (Step Size)
        numerator = np.dot(r, r)
        denominator = np.dot(r, Ar)

        # Avoid division by zero
        if denominator == 0:
            break

        alpha = numerator / denominator

        # x(k) = x(k-1) + alpha * r(k-1)
        x = x + alpha * r

        # r(k) = r(k-1) - alpha * A * r(k-1)
        r = r - alpha * Ar

    return x, itmax


def solve_pcgm(matrix, b, P, x0, itmax, TOL):
    matrix = np.array(matrix, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    P = np.array(P, dtype=float)

    # r(0) = b - Ax
    r = b - np.dot(matrix, x)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0

    # z(0): solve P z = r
    z = np.linalg.solve(P, r)

    # rho = r^T z
    rho_prev = np.dot(r, z)

    # p(0) = z(0)
    p = z.copy()

    for k in range(itmax):
        # Check convergence (normalized residue)
        r_norm = np.linalg.norm(r)
        if r_norm / norm_b < TOL:
            return x, k

        # Matrix * p
        Ap = np.dot(matrix, p)

        # Alpha = (r^T z) / (p^T A p)
        denom = np.dot(p, Ap)
        if denom == 0:
            break

        alpha = rho_prev / denom

        # x = x + alpha * p
        x = x + alpha * p

        # r = r - alpha * A * p
        r = r - alpha * Ap

        # z: solve P z = r
        z = np.linalg.solve(P, r)

        # Beta = (r_new^T z_new) / (r_old^T z_old)
        rho_curr = np.dot(r, z)
        if rho_prev == 0:
            break

        beta = rho_curr / rho_prev

        # p = z + beta * p
        p = z + beta * p

        rho_prev = rho_curr

    return x, itmax


def solve_cgm(matrix, b, x0, itmax, TOL):
    matrix = np.array(matrix, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    k = 0

    # r = b - A * x
    r = b - np.dot(matrix, x)

    # spectral(new) = r(transpose) * r
    spectral_new = np.dot(r, r)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0

    # while ||r|| > TOL * ||b||
    while np.sqrt(spectral_new) / norm_b > TOL and k < itmax:
        k += 1

        # Calculate Direction p
        if k == 1:
            p = r.copy()
        else:
            # beta = spectral(new) / spectral(old)
            if spectral_old == 0:
                break
            beta = spectral_new / spectral_old

            # p = r + beta * p
            p = r + beta * p

        # w = A * p
        w = np.dot(matrix, p)

        # alpha = spectral(new) / (p(transpose) * w)
        denom = np.dot(p, w)
        if denom == 0:
            break
        alpha = spectral_new / denom

        # x = x + alpha * p
        x = x + alpha * p

        # r = r - alpha * w
        r = r - alpha * w

        # spectral(old) = spectral(new)
        spectral_old = spectral_new

        # spectral(new) = r(transpose) * r
        spectral_new = np.dot(r, r)

    return x, k


def solve_cgnr(matrix, b, x0, itmax, TOL):
    matrix = np.array(matrix, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    # CGNR solves (A^T * A) x = A^T * b

    # r = b - A * x
    r = b - np.dot(matrix, x)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0

    # z = A^T * r
    z = np.dot(matrix.T, r)

    # p = z
    p = z.copy()

    for k in range(itmax):
        r_norm = np.linalg.norm(r)

        # Stopping condition: normalized residue
        if r_norm / norm_b < TOL:
            return x, k

        # w = A * p
        w = np.dot(matrix, p)

        # Alpha: (z . z) / (w . w)
        z_dot_z = np.dot(z, z)
        w_dot_w = np.dot(w, w)
        if w_dot_w == 0:
            break

        alpha = z_dot_z / w_dot_w

        # Update x: x = x + alpha * p
        x = x + alpha * p

        # Update r: r = r - alpha * w
        r = r - alpha * w

        # Update z_new: z_new = A^T * r_new
        z_new = np.dot(matrix.T, r)

        # Beta: (z_new . z_new) / (z . z)
        z_new_dot = np.dot(z_new, z_new)
        if z_dot_z == 0:
            break
        beta = z_new_dot / z_dot_z

        # Update p: p = z_new + beta * p
        p = z_new + beta * p

        # Update z for next iteration
        z = z_new

    return x, itmax


def solve_cgne(matrix, b, x0, itmax, TOL):
    matrix = np.array(matrix, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    # CGNE solves A * A^T * y = b, where x = A^T * y

    # r = b - A * x
    r = b - np.dot(matrix, x)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0

    # p = A^T * r
    p = np.dot(matrix.T, r)

    for k in range(itmax):
        r_norm = np.linalg.norm(r)

        # Stopping condition: normalized residue
        if r_norm / norm_b < TOL:
            return x, k

        # Alpha: (r . r) / (p . p)
        r_dot_r = np.dot(r, r)
        p_dot_p = np.dot(p, p)
        if p_dot_p == 0:
            break

        alpha = r_dot_r / p_dot_p

        # Update x: x = x + alpha * p
        x = x + alpha * p

        # Update r: r = r - alpha * A * p
        Ap = np.dot(matrix, p)
        r = r - alpha * Ap

        # Beta: (r_new . r_new) / (r_old . r_old)
        r_new_dot = np.dot(r, r)
        if r_dot_r == 0:
            break
        beta = r_new_dot / r_dot_r

        # Update p: p = A^T * r + beta * p
        At_r = np.dot(matrix.T, r)
        p = At_r + beta * p

    return x, itmax


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
    x0 = np.zeros(5)

    itmax = 1000
    TOL = 1e-10

    x, kf = solve_cgnr(matrix, b, x0, itmax, TOL)

    print(f"Final Solution x (CGNR): {np.round(x, 6)}")
    print(f"Iterations: {kf}")


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
    x0 = np.zeros(5)

    itmax = 1000
    TOL = 1e-10

    x, kf = solve_cgne(matrix, b, x0, itmax, TOL)

    print(f"Final Solution x (CGNE): {np.round(x, 6)}")
    print(f"Iterations: {kf}")


def run():
    matrix = np.array([
        [60.0, 10.0, 5.0, 2.0, 1.0],
        [10.0, 55.0, 4.0, 2.0, 1.0],
        [5.0, 4.0, 50.0, 6.0, 2.0],
        [2.0, 2.0, 6.0, 45.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 40.0]
    ])

    b = np.array([264.0, 289.0, 331.0, 335.0, 315.0])
    x0 = np.zeros(5)

    itmax = 1000
    TOL = 1e-10

    # SDM
    x_sdm, kf_sdm = solve_steepest_descent(matrix, b, x0, itmax, TOL)
    print("-" * 50)
    print(f"SDM: Solution = {np.round(x_sdm, 6)} | Iterations = {kf_sdm}")

    # CGM
    x_cgm, kf_cgm = solve_cgm(matrix, b, x0, itmax, TOL)
    print(f"CGM: Solution = {np.round(x_cgm, 6)} | Iterations = {kf_cgm}")

    # PCGM (Jacobi preconditioner: diagonal of A)
    P = np.diag(np.diag(matrix))
    x_pcgm, kf_pcgm = solve_pcgm(matrix, b, P, x0, itmax, TOL)
    print(f"PCGM: Solution = {np.round(x_pcgm, 6)} | Iterations = {kf_pcgm} (P=diag(A))")
    print("-" * 50)


# Redo a lot of stuff for exercise 6

def u_exact_1d(x, ell, m):
    # u(ex)(x) = x^ell (1-x)^m
    return (x ** ell) * ((1.0 - x) ** m)

def u2_exact_1d(x, ell, m):
    # u''(x) for u = x^ell (1-x)^m
    t1 = ell * (ell - 1) * (x ** (ell - 2)) * ((1.0 - x) ** m)
    t2 = 2.0 * ell * m * (x ** (ell - 1)) * ((1.0 - x) ** (m - 1))
    t3 = m * (m - 1) * (x ** ell) * ((1.0 - x) ** (m - 2))
    return t1 - t2 + t3

def f_rhs_1d(x, ell, m):
    # f(x) = -u''(x) + u(x)
    return -u2_exact_1d(x, ell, m) + u_exact_1d(x, ell, m)

def build_fdm_system_1d(n, ell, m):
    # Grid: x_i = i h, i=0..n+1, h=1/(n+1)
    h = 1.0 / (n + 1)
    x_int = np.arange(1, n + 1, dtype=float) * h

    # A for -u'' + u with Dirichlet boundaries (0 at ends)
    diag = (2.0 / (h * h) + 1.0) * np.ones(n)
    off = (-1.0 / (h * h)) * np.ones(n - 1)
    A = np.diag(diag) + np.diag(off, k=-1) + np.diag(off, k=1)

    # RHS and exact interior solution
    f = f_rhs_1d(x_int, ell, m)
    uex = u_exact_1d(x_int, ell, m)

    return A, f, uex


def apply_precond_jacobi(diagA, r):
    # z = P^{-1} r, P = diag(A)
    return r / diagA

def apply_precond_rownorm2(row_norm, r):
    # z = P^{-1} r, P = diag(||row_i||_2)
    return r / row_norm

def ssor_solve_bidiagonal(diagA, offA, rhs, omega, lower=True):
    # Solve (D + omega*L) y = rhs or (D + omega*U) y = rhs (tridiagonal case)
    n = diagA.size
    y = np.zeros(n, dtype=float)

    if lower:
        y[0] = rhs[0] / diagA[0]
        for i in range(1, n):
            y[i] = (rhs[i] - omega * offA[i - 1] * y[i - 1]) / diagA[i]
    else:
        y[-1] = rhs[-1] / diagA[-1]
        for i in range(n - 2, -1, -1):
            y[i] = (rhs[i] - omega * offA[i] * y[i + 1]) / diagA[i]

    return y

def apply_precond_ssor_scaled(diagA, offA, r, omega):
    # P = omega(2-omega) * (D + omega L) D^{-1} (D + omega U)
    scale = omega * (2.0 - omega)
    rhs = r / scale

    y = ssor_solve_bidiagonal(diagA, offA, rhs, omega, lower=True)
    y = y / diagA
    z = ssor_solve_bidiagonal(diagA, offA, y, omega, lower=False)

    return z


def solve_pcgm_ex6(matrix, b, x0, itmax, TOL, precond_type="jacobi", omega=1.0):
    matrix = np.array(matrix, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    # r(0) = b - Ax
    r = b - np.dot(matrix, x)

    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0

    diagA = np.diag(matrix).copy()
    offA = np.diag(matrix, k=1).copy()

    # z(0) = P^{-1} r(0)
    if precond_type == "jacobi":
        z = apply_precond_jacobi(diagA, r)
    elif precond_type == "rownorm2":
        row_norm = np.linalg.norm(matrix, axis=1)
        z = apply_precond_rownorm2(row_norm, r)
    elif precond_type == "ssor":
        z = apply_precond_ssor_scaled(diagA, offA, r, omega)
    else:
        raise ValueError("Unknown precond_type")

    # rho = r^T z
    rho_prev = np.dot(r, z)

    # p(0) = z(0)
    p = z.copy()

    for k in range(itmax):
        # Check convergence (normalized residue)
        r_norm = np.linalg.norm(r)
        if r_norm / norm_b < TOL:
            return x, k

        Ap = np.dot(matrix, p)

        denom = np.dot(p, Ap)
        if denom == 0:
            break

        alpha = rho_prev / denom

        x = x + alpha * p
        r = r - alpha * Ap

        if precond_type == "jacobi":
            z = apply_precond_jacobi(diagA, r)
        elif precond_type == "rownorm2":
            z = apply_precond_rownorm2(row_norm, r)
        else:
            z = apply_precond_ssor_scaled(diagA, offA, r, omega)

        rho_curr = np.dot(r, z)
        if rho_prev == 0:
            break

        beta = rho_curr / rho_prev
        p = z + beta * p
        rho_prev = rho_curr

    return x, itmax


def run_ex6():
    print("-" * 50)
    print("Exercise 6: FDM 1D (-u'' + u)")

    cases = [(2, 2), (10, 2), (2, 10)]
    n_list = [49, 99, 199, 399, 799]

    itmax = 50000
    TOL = 1e-10
    omega = 1.0

    # (c) CGM
    print("-" * 50)
    print("Part (c): CGM")
    for ell, m in cases:
        print("-" * 50)
        print(f"(ell, m) = ({ell}, {m})")
        print("n    | kf    | ||e||_inf")
        print("-" * 28)

        for n in n_list:
            A, f, uex = build_fdm_system_1d(n, ell, m)
            x0 = np.zeros(n)

            uh, kf = solve_cgm(A, f, x0, itmax, TOL)
            err_inf = np.linalg.norm(uh - uex, ord=np.inf)

            print(f"{n:<4} | {kf:<5} | {err_inf:.3e}")

    # (d) PCGM with different P
    preconds = [("PJ", "jacobi"), ("P_row2", "rownorm2"), ("P_SSOR_scaled", "ssor")]

    print("-" * 50)
    print("Part (d): PCGM")
    for ell, m in cases:
        print("-" * 50)
        print(f"(ell, m) = ({ell}, {m})")

        for label, ptype in preconds:
            if ptype == "ssor":
                print("-" * 50)
                print(f"Precond: {label} (omega={omega})")
            else:
                print("-" * 50)
                print(f"Precond: {label}")

            print("n    | kf    | ||e||_inf")
            print("-" * 28)

            for n in n_list:
                A, f, uex = build_fdm_system_1d(n, ell, m)
                x0 = np.zeros(n)

                uh, kf = solve_pcgm_ex6(A, f, x0, itmax, TOL, precond_type=ptype, omega=omega)
                err_inf = np.linalg.norm(uh - uex, ord=np.inf)

                print(f"{n:<4} | {kf:<5} | {err_inf:.3e}")

if __name__ == "__main__":
    run()
    run_ex4()
    run_ex5()
    run_ex6()
