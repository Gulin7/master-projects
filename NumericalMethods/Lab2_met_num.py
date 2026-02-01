import numpy as np

def steepest_descent(A, b, x0, itmax, TOL):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    

    r = b - np.dot(A, x)
    
    norm_b = np.linalg.norm(b, ord=2)
    
    if norm_b == 0:
        return x, 0

    for k in range(itmax):
        norm_r = np.linalg.norm(r, ord=2)
        
        if norm_r / norm_b < TOL:
            kf = k 
            return x, kf
        
        Ar = np.dot(A, r)
        
        rTr = np.dot(r, r)
        rAr = np.dot(r, Ar)
        
        if rAr == 0:
            break
            
        alpha = rTr / rAr
        
        x = x + alpha * r
        r = r - alpha * Ar

    return x, itmax

def CGM(A, b, x0, itmax, TOL):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    
    # 1. Inițializare
    r = b - np.dot(A, x) 
    p = r
    
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0
    
    # r_k^T * r_k (utilizat pentru beta)
    rho_prev = np.dot(r, r)

    for k in range(itmax):
        # Criteriul de oprire: Eroarea reziduală normalizată
        if np.linalg.norm(r) / norm_b < TOL:
            kf = k
            return x, kf
        
        # Calculăm A * p_k
        Ap = np.dot(A, p)
        
        # 2. Calculăm pasul optim alpha_k
        # alpha = (r_k^T * r_k) / (p_k^T * A * p_k)
        alpha = rho_prev / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap
        
        rho_curr = np.dot(r, r)
        beta = rho_curr / rho_prev
        

        p = r + beta * p
    
        rho_prev = rho_curr

    return x, itmax



def PCGM(A, b, P, x0, itmax, TOL):
    # Aici presupunem că P este o funcție care rezolvă Mz = r
    # Dacă P este o matrice, rezolvăm: Mz = r, unde M = P
    
    # Notă: Matricea P ar trebui să fie de fapt o metodă (inversa precondiționerului M)
    # În exemplul simplu, P este inversul precondiționerului M, M^{-1}
    # Dacă P este chiar matricea precondiționer M, trebuie rezolvat sistemul Mz = r
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    P = np.array(P, dtype=float) # Presupunem că P este M^{-1} sau trebuie rezolvat Mz = r
    
    # 1. Inițializare
    r = b - np.dot(A, x)  # Reziduul inițial
    
    # Rezolvăm M z = r pentru a obține z (reziduul precondiționat)
    # Aici folosim np.linalg.solve(P, r) presupunând că P este matricea precondiționer M
    z = np.linalg.solve(P, r) # Operație specifică PCGM: M * z = r
    
    p = z                 # Direcția inițială p este egală cu z
    
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return x, 0
    
    # r_k^T * z_k (utilizat pentru alpha și beta)
    rho_prev = np.dot(r, z)

    for k in range(itmax):
        # Criteriul de oprire: Eroarea reziduală normalizată
        if np.linalg.norm(r) / norm_b < TOL:
            kf = k
            return x, kf
        
        # Calculăm A * p_k
        Ap = np.dot(A, p)
        
        # 2. Calculăm pasul optim alpha_k
        # alpha = (r_k^T * z_k) / (p_k^T * A * p_k)
        alpha = rho_prev / np.dot(p, Ap)
        
        # 3. Actualizăm soluția și reziduul
        x = x + alpha * p
        r = r - alpha * Ap
        
        # 4. Rezolvăm M z_{k+1} = r_{k+1}
        z = np.linalg.solve(P, r) # Operație specifică PCGM: M * z = r
        
        # 5. Calculăm noul beta
        rho_curr = np.dot(r, z)   # Noul rho este r_{k+1}^T * z_{k+1}
        beta = rho_curr / rho_prev
        
        # 6. Actualizăm direcția de căutare (conjugată)
        # p_{k+1} = z_{k+1} + beta_k * p_k
        p = z + beta * p
        
        # Pregătire pentru următoarea iterație
        rho_prev = rho_curr

    return x, itmax

if __name__ == "__main__":
    A_test = np.array([[60, 10, 5, 2, 1],
              [10, 55, 4, 2, 1],
              [5, 4, 50, 6, 2],
              [2, 2, 6, 45, 3],
              [1, 1, 2, 3, 40]], dtype=float)
    
    b_test = np.array([264, 289, 331, 335, 315], dtype=float)
    x0_test = np.array([0, 0, 0, 0, 0], dtype=float)
    
    itmax = 1000
    TOL = 1e-10

    # Pentru PCGM, vom folosi ca precondiționer M, matricea diagonală a lui A (Jacobi Preconditioner)
    # P = Matricea precondiționer M (Diagonala lui A)
    P_diag = np.diag(np.diag(A_test))
# Setăm opțiunile de printare pentru o vizualizare mai clară
    np.set_printoptions(precision=4, suppress=True)

    print("-" * 50)

    # Rulare SDM
    x_sdm, kf_sdm = steepest_descent(A_test, b_test, x0_test, itmax, TOL)
    x_formatted = [format(val, '.6f') for val in x_sdm]
    print(f"SDM: Solutia = {x_formatted} | Iteratii = {kf_sdm}")
    
    # Rulare CGM
    x_cgm, kf_cgm = CGM(A_test, b_test, x0_test, itmax, TOL)
    x_formatted = [format(val, '.6f') for val in x_cgm]
    print(f"CGM: Solutia = {x_formatted} | Iteratii = {kf_cgm}")

 
    x_pcgm, kf_pcgm = PCGM(A_test, b_test, P_diag, x0_test, itmax, TOL)
    x_formatted = [format(val, '.6f') for val in x_pcgm]
    print(f"PCGM: Solutia = {x_formatted} | Iteratii = {kf_pcgm} (P=Diagonala lui A)")
    print("-" * 50)