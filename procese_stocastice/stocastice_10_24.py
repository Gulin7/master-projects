import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parametrii
n = 40          # n variabile
m = 10000       # m simulări
a, b = -1, 2    # Intervalul pentru care calculăm frecvența relativă

# distribuție Uniformă între 0 și 1
# Pentru Uniform(0,1): media mu = 0.5, varianța sigma^2 = 1/12
mu = 0.5
sigma_sq = 1/12
sigma = np.sqrt(sigma_sq)

# Generăm datele (o matrice de m linii și n coloane)
# Fiecare linie este un experiment cu n variabile
date = np.random.uniform(0, 1, (m, n))

# Calculăm suma pe fiecare linie
suma_xk = np.sum(date, axis=1)

# Calculăm S_n hat (Standardizarea)
# Aplicăm formula din cerință pentru toate cele m simulări simultan
s_n_hat = (suma_xk - n * mu) / np.sqrt(n * sigma_sq)

# Calculăm frecvența relativă în intervalul [a, b]
# Numărăm câte valori din s_n_hat sunt între -1 și 2, apoi împărțim la m
frecventa_relativa = np.sum((s_n_hat >= a) & (s_n_hat <= b)) / m

# Calculăm valoarea teoretică folosind distribuția Normală Standard
# P(a <= Z <= b) = Phi(b) - Phi(a)
valoare_teoretica = norm.cdf(b) - norm.cdf(a)

print(f"Frecvența relativă simulată: {frecventa_relativa:.4f}")
print(f"Valoarea teoretică (Normală): {valoare_teoretica:.4f}")

# Plotting - Vizualizarea rezultatelor
plt.hist(s_n_hat, bins=50, density=True, alpha=0.6, color='skyblue', label='Simulare Sn_hat')

# Curba Normală Standard (0, 1) pentru comparație
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label='Normala Standard (PDF)')

plt.title(f'Distribuția sumei standardizate (n={n}, m={m})')
plt.axvspan(a, b, color='green', alpha=0.2, label=f'Interval [{a}, {b}]')
plt.legend()
plt.show()