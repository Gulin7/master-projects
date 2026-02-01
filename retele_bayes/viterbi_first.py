import numpy as np

# ==========================================
# 1. ALGORITMUL VITERBI (GENERALIZAT)
# ==========================================
def algoritm_viterbi_generic(obs, start_p, trans_p, emit_p):
    """
    Algoritmul Viterbi generic.
    Nu depinde de nicio variabila globala.
    """
    n_states = trans_p.shape[0]
    T = len(obs)
    
    # Initializare tabele
    v_table = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)
    
    # Pasul de initializare (t=0)
    for s in range(n_states):
        v_table[s, 0] = start_p[s] * emit_p[s, obs[0]]
        
    # Pasul de recursivitate (t > 0)
    for t in range(1, T):
        for s in range(n_states):
            trans_probs = [v_table[s_prev, t-1] * trans_p[s_prev, s] * emit_p[s, obs[t]] 
                           for s_prev in range(n_states)]
            
            v_table[s, t] = np.max(trans_probs)
            backpointer[s, t] = np.argmax(trans_probs)
            
    # Terminarea si Backtracking
    best_last_state = np.argmax(v_table[:, T-1])
    best_path = [best_last_state]
    
    for t in range(T-1, 0, -1):
        prev_state = backpointer[best_path[-1], t]
        best_path.append(prev_state)
        
    best_path.reverse()
    return best_path

# ==========================================
# 2. SCENARIUL 1: PREDICTIA VREMII
# ==========================================

def get_params_predictia_vremii():
    """
    Aceasta functie contine TOATE configuratiile specifice problemei vremii.
    Returneaza tot ce e nevoie pentru a rula simularea.
    """
    states = ["Cald", "Rece"]
    obs_vocab = ["1 inghetata", "2 inghetate", "3 inghetate"]

    # Probabilitati
    start_p = np.array([0.8, 0.2])
    
    trans_p = np.array([
        [0.7, 0.3], # Cald -> Cald/Rece
        [0.4, 0.6]  # Rece -> Cald/Rece
    ])
    
    emit_p = np.array([
        [0.1, 0.4, 0.5], # Cald -> 1, 2, 3 ingh
        [0.7, 0.2, 0.1]  # Rece -> 1, 2, 3 ingh
    ])

    return states, obs_vocab, start_p, trans_p, emit_p

def genereaza_date_generic(n_zile, start_p, trans_p, emit_p):
    """
    Genereaza date pe baza matricilor primite ca argument.
    Nu mai depinde de variabile globale.
    """
    stari_reale = []
    observatii = []
    
    # Numarul de stari si observatii se deduce din dimensiunile matricilor
    n_states = len(start_p)
    n_obs = emit_p.shape[1]
    
    stare_curenta = np.random.choice(range(n_states), p=start_p)
    
    for _ in range(n_zile):
        stari_reale.append(stare_curenta)
        
        # Generam observatia
        obs = np.random.choice(range(n_obs), p=emit_p[stare_curenta])
        observatii.append(obs)
        
        # Generam tranzitia
        stare_curenta = np.random.choice(range(n_states), p=trans_p[stare_curenta])
        
    return stari_reale, observatii

def run_predictia_vremii():
    print("\n--- SCENARIU: PREDICTIA VREMII ---")
    
    # 1. Incarcam parametrii locali (fara globale!)
    states, obs_vocab, start_p, trans_p, emit_p = get_params_predictia_vremii()

    # Input validation
    while True:
        try:
            n_input = input("Introduceti numarul de zile pentru simulare (ex: 10): ")
            n_zile = int(n_input)
            if n_zile <= 0:
                print("Te rog introdu un numar pozitiv.")
                continue
            break
        except ValueError:
            print("Eroare: Te rog introdu un numar intreg valid.")

    # 2. Generare date (trimitem parametrii explicit)
    x_real, y_obs = genereaza_date_generic(n_zile, start_p, trans_p, emit_p)
    
    # 3. Rulare Viterbi
    x_estimat = algoritm_viterbi_generic(y_obs, start_p, trans_p, emit_p)
    
    # 4. Afisare rezultate
    print(f"\n{'ZI':<5} {'OBSERVATIE':<20} {'REAL':<10} {'VITERBI':<10} {'STATUS'}")
    print("-" * 65)

    corecte = 0
    for i in range(n_zile):
        status = "OK" if x_real[i] == x_estimat[i] else "X"
        if status == "OK": corecte += 1
        
        obs_str = obs_vocab[y_obs[i]]
        real_str = states[x_real[i]]
        est_str = states[x_estimat[i]]
        
        print(f"{i:<5} {obs_str:<20} {real_str:<10} {est_str:<10} {status}")

    acuratete = (corecte / n_zile) * 100
    print("-" * 65)
    print(f"Rezultat: {corecte}/{n_zile} corecte. Acuratete: {acuratete:.2f}%")
    
    return n_zile, acuratete

# ==========================================
# 3. MAIN LOOP
# ==========================================

def run():
    istoric_rezultate = [] 

    print("=== PROIECT HMM VITERBI ===")

    while True:
        print("\nMENIU PRINCIPAL:")
        print("1 - Predictia Vremii (Jason Eisner)")
        print("0 - Exit")
        
        optiune = input("\nSelectati o optiune: ")
        
        if optiune == '0':
            print("Iesire din program...")
            break
            
        elif optiune == '1':
            rezultat = run_predictia_vremii()
            if rezultat:
                zile, acc = rezultat
                istoric_rezultate.append((1, zile, acc))
                input("\nApasa Enter pentru a reveni la meniu...")
        
        else:
            print("Optiune invalida.")

    # Raport final
    if istoric_rezultate:
        print("\n" + "="*40)
        print("RAPORT FINAL SESIUNE")
        print("="*40)
        print(f"{'ID':<5} {'PROBLEMA':<20} {'ZILE':<10} {'ACURATETE'}")
        print("-" * 40)
        
        nume_probleme = {1: "Predictia Vremii"}
        
        for record in istoric_rezultate:
            p_id, p_zile, p_acc = record
            nume = nume_probleme.get(p_id, "Necunoscut")
            print(f"{p_id:<5} {nume:<20} {p_zile:<10} {p_acc:.2f}%")
        print("="*40)

if __name__ == '__main__':
    run()