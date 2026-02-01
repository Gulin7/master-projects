import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Se descarcă datele...")
tickers = ['GOOGL', 'SPY']
start_date = '2020-01-01'
end_date = '2024-01-01'

# Descărcăm datele
raw_data = yf.download(tickers, start=start_date, end=end_date)

# Inițializăm variabila data cu None
data = None
if raw_data is not None and not raw_data.empty:
    
    # Acum e sigur să verificăm coloanele
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
        print("Am folosit coloana: Adj Close")
    elif 'Close' in raw_data.columns:
        data = raw_data['Close']
        print("Adj Close nu există. Am folosit coloana: Close")
    else:
        print("Eroare: Nu găsesc nici coloana Adj Close, nici Close.")
        exit()
        
    print("Date descărcate cu succes!")
    print(data.head())

else:
    print("Eroare: Nu s-au putut descărca datele. Verifică internetul sau ticker-ii.")
    exit() # Oprim scriptul aici dacă nu avem date

# Formula: ln(Pt / Pt-1)
# Calculăm logaritmii
# Folosim .apply(np.log) pentru a evita erorile de tip Pylance
log_returns = (data / data.shift(1)).apply(np.log)

# Curățăm datele lipsă
log_returns = pd.DataFrame(log_returns).dropna()

r_googl = log_returns['GOOGL']
r_spy = log_returns['SPY']

def analyze_horizon(series_x, series_y, name_x, name_y, horizon_label):
    # Definim intervalele (bins) bazate pe quantile (20%, 40%, 60%, 80%)
    # Acest lucru împarte datele în 5 categorii egale numeric
    bins_x = np.quantile(series_x, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins_y = np.quantile(series_y, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 2. Creăm histograma comună (Joint Histogram)
    joint_counts, _, _ = np.histogram2d(series_x, series_y, bins=[bins_x, bins_y])
    
    # Inversăm axa Y pentru vizualizare (ca graficul să aibă originea jos-stânga)
    joint_counts = np.flipud(joint_counts).T
    
    # Calculăm valorile așteptate (dacă ar fi independente)
    total_obs = np.sum(joint_counts)
    marginal_x = np.sum(joint_counts, axis=1)
    marginal_y = np.sum(joint_counts, axis=0)
    expected_counts = np.outer(marginal_y, marginal_x) / total_obs # Atenție la ordinea axelor
    
    # --- Vizualizare ---
    plt.figure(figsize=(14, 5))
    
    # Grafic A: Histograma Comună (Frecvențe reale)
    plt.subplot(1, 2, 1)
    sns.heatmap(joint_counts, annot=True, fmt='.0f', cmap="Blues", cbar=False)
    plt.title(f'{horizon_label}: Joint Histogram ({name_x} vs {name_y})')
    plt.xlabel(f'{name_x} Returns (Quantiles)')
    plt.ylabel(f'{name_y} Returns (Quantiles)')
    
    # Grafic B: Diferența (Real - Teoretic Independent)
    # Roșu = Mai multe observații decât ne așteptam (Dependență)
    # Albastru = Mai puține observații
    plt.subplot(1, 2, 2)
    difference = joint_counts - expected_counts
    sns.heatmap(difference, annot=True, fmt='.1f', cmap="coolwarm", center=0)
    plt.title(f'{horizon_label}: Diferența (Observat - Independent)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"--- Concluzie sumară pentru {horizon_label} ---")
    print(f"Număr total observații: {int(total_obs)}")

# Full Sample (Tot istoricul descărcat)
analyze_horizon(r_spy, r_googl, "SPY (Market)", "GOOGL", "Full Sample")

# 6 Luni (Ultimele aprox. 126 zile de tranzacționare)
analyze_horizon(r_spy.tail(126), r_googl.tail(126), "SPY (Market)", "GOOGL", "6 Months")

# 1 Lună (Ultimele aprox. 21 zile de tranzacționare)
analyze_horizon(r_spy.tail(21), r_googl.tail(21), "SPY (Market)", "GOOGL", "1 Month")