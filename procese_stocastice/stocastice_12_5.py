import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Definirea Matricei și a Stărilor
states = ['L', 'M', 'H']
# Ordinea: L (0), M (1), H (2)
P = np.array([
    [0.2, 0.6, 0.2],  # Din L
    [0.1, 0.7, 0.2],  # Din M
    [0.05, 0.15, 0.8] # Din H
])

def prob_reaching_L_within_k_days(matrix, start_state_idx, k_days):
    """
    Folosim metoda stării absorbante. Modificăm matricea astfel încât
    odată intrat în L (index 0), sistemul rămâne în L.
    """
    P_abs = matrix.copy()
    # Facem L stare absorbantă: prob 1 sa ramai in L, 0 sa pleci
    P_abs[0] = [1.0, 0.0, 0.0] 
    
    # Starea inițială (vector linie): 0 pentru L, 0 pentru M, 1 pentru H
    initial_dist = np.zeros(3)
    initial_dist[start_state_idx] = 1.0
    
    # Calculăm distribuția după k pași: v_k = v_0 * P^k
    final_dist = initial_dist @ np.linalg.matrix_power(P_abs, k_days)
    
    # Probabilitatea de a fi în L după k pași (în matricea modificată)
    # este echivalentă cu probabilitatea de a fi atins L oricând în acei k pași.
    return final_dist[0]

# Presupunem că începem cu stoc plin (H - index 2)
prob_fail = prob_reaching_L_within_k_days(P, 2, 7)
print(f"Probabilitatea de a ajunge în starea L (stoc epuizat) într-o săptămână (start H): {prob_fail:.4f}")

# Metoda A: Numerică (Ridicarea matricei la putere mare)
P_limit = np.linalg.matrix_power(P, 1000)
stationary_numerical = P_limit[0] # Toate rândurile vor fi identice

# Metoda B: Teoretică (Valori proprii / Eigenvalues)
# Căutăm vectorul v pentru care vP = v <=> P.T * v.T = v.T
eig_vals, eig_vecs = np.linalg.eig(P.T)

# Găsim indexul valorii proprii care este (aproape) 1
idx = np.argmin(np.abs(eig_vals - 1.0))
stationary_theoretical = np.real(eig_vecs[:, idx])
# Normalizăm vectorul ca suma să fie 1
stationary_theoretical /= stationary_theoretical.sum()

print("\n--- Distribuția Staționară ---")
print(f"Teoretică (Eigen): L={stationary_theoretical[0]:.4f}, M={stationary_theoretical[1]:.4f}, H={stationary_theoretical[2]:.4f}")
print(f"Numerică (P^1000): L={stationary_numerical[0]:.4f}, M={stationary_numerical[1]:.4f}, H={stationary_numerical[2]:.4f}")

def simulate_inventory(matrix, steps, start_idx):
    current_state = start_idx
    counts = {0: 0, 1: 0, 2: 0} # 0:L, 1:M, 2:H
    
    for _ in range(steps):
        # Alegem următoarea stare bazat pe probabilitățile rândului curent
        probs = matrix[current_state]
        current_state = np.random.choice([0, 1, 2], p=probs)
        counts[current_state] += 1
        
    # Calculăm probabilitățile empirice
    return [counts[0]/steps, counts[1]/steps, counts[2]/steps]

n_steps = 100_000
empirical_dist = simulate_inventory(P, n_steps, 2) # Start in H

print(f"\n--- Simulare ({n_steps} pași) ---")
print(f"Empirică: L={empirical_dist[0]:.4f}, M={empirical_dist[1]:.4f}, H={empirical_dist[2]:.4f}")

labels = ['Low (L)', 'Medium (M)', 'High (H)']
x = np.arange(len(labels))  # locațiile etichetelor
width = 0.25  # lățimea barelor

fig, ax = plt.subplots(figsize=(10, 6))

# Crearea barelor grupate
rects1 = ax.bar(x - width, stationary_theoretical, width, label='Teoretic (Eigen)', color='navy', alpha=0.8)
rects2 = ax.bar(x, stationary_numerical, width, label='Numeric (P^1000)', color='skyblue', alpha=0.8)
rects3 = ax.bar(x + width, empirical_dist, width, label=f'Empiric (Simulare {n_steps})', color='darkorange', alpha=0.8)

# Adăugare text, titlu și etichete
ax.set_ylabel('Probabilitate')
ax.set_title('Comparație Distribuție Staționară: Teoretic vs Numeric vs Simulat')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1.0) # Setăm limita Y până la 1 pentru claritate
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Funcție pentru a pune etichete cu valori deasupra barelor
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects3)

plt.tight_layout()
plt.show()