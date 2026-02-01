import numpy as np
import os

# ==========================================
# 0. SETUP AUTOMAT (Fisier Date Antrenament)
# ==========================================
def asigura_existenta_training_data():
    filename = 'training_data.txt'
    if not os.path.exists(filename):
        print(f"Generare fisier '{filename}' pentru antrenare...")
        text_demo = """
        Algoritmul Viterbi este o metoda de programare dinamica.
        Vremea este frumoasa si am mancat o inghetata buna.
        Recunoasterea textului este o problema complexa de inteligenta artificiala.
        Ana are mere si pere. Studentii invata la facultate despre probabilitati.
        Acesta este un text de test pentru modelul de limba romana.
        Un alt exemplu pentru a avea mai multe date de tranzitie intre litere.
        Muntii Carpati sunt o destinatie preferata pentru drumetii si sporturi de iarna.
        Marea Neagra atrage turisti in fiecare vara in statiuni precum Mamaia si Vama Veche.
        Romania are un relief variat care include munti dealuri campii si delta dunarii.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_demo)

# ==========================================
# 1. ALGORITMI (VITERBI & GREEDY)
# ==========================================
def algoritm_viterbi_generic(obs, start_p, trans_p, emit_p):
    """ Gaseste calea cea mai probabila global (contextual). """
    n_states = trans_p.shape[0]
    T = len(obs)
    
    v_table = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)
    
    # Initializare
    for s in range(n_states):
        v_table[s, 0] = start_p[s] * emit_p[s, obs[0]]
        
    # Recursivitate
    for t in range(1, T):
        for s in range(n_states):
            trans_probs = [v_table[s_prev, t-1] * trans_p[s_prev, s] * emit_p[s, obs[t]] 
                           for s_prev in range(n_states)]
            v_table[s, t] = np.max(trans_probs)
            backpointer[s, t] = np.argmax(trans_probs)
            
    # Backtracking
    best_last_state = np.argmax(v_table[:, T-1])
    best_path = [best_last_state]
    
    for t in range(T-1, 0, -1):
        prev_state = backpointer[best_path[-1], t]
        best_path.append(prev_state)
        
    best_path.reverse()
    return best_path

def algoritm_greedy_naiv(obs, emit_p):
    """ 
    Alege starea cea mai probabila strict pe baza observatiei curente.
    Ignora tranzitiile (istoricul).
    """
    path = []
    for o in obs:
        # Care stare (rand) are cea mai mare valoare pe coloana observatiei 'o'?
        best_state_now = np.argmax(emit_p[:, o])
        path.append(best_state_now)
    return path

def genereaza_date_generic(n_steps, start_p, trans_p, emit_p):
    stari_reale = []
    observatii = []
    
    n_states = len(start_p)
    n_obs = emit_p.shape[1]
    
    stare_curenta = np.random.choice(range(n_states), p=start_p)
    
    for _ in range(n_steps):
        stari_reale.append(stare_curenta)
        obs = np.random.choice(range(n_obs), p=emit_p[stare_curenta])
        observatii.append(obs)
        stare_curenta = np.random.choice(range(n_states), p=trans_p[stare_curenta])
        
    return stari_reale, observatii

# ==========================================
# 2. SCENARIUL VREME (PARAMETRI)
# ==========================================

def get_params_vreme_standard():
    """ 2 Stari: Cald, Rece (Modelul Clasic Eisner) """
    states = ["Cald", "Rece"]
    obs_vocab = ["1 ingh", "2 ingh", "3 ingh"]

    start_p = np.array([0.8, 0.2])
    
    # Tranzitie: Inertie moderata
    trans_p = np.array([
        [0.7, 0.3], 
        [0.4, 0.6]  
    ])
    
    # Emisie: Destul de clara (1->Rece, 3->Cald)
    emit_p = np.array([
        [0.1, 0.4, 0.5], 
        [0.7, 0.2, 0.1]  
    ])
    return states, obs_vocab, start_p, trans_p, emit_p

def get_params_vreme_complex():
    """ 4 Stari: F.Cald, Cald, Rece, F.Rece """
    states = ["F. Cald", "Cald", "Rece", "F. Rece"]
    obs_vocab = ["1 ingh", "2 ingh", "3 ingh"]

    start_p = np.array([0.4, 0.4, 0.1, 0.1])
    
    # Tranzitie: Graduala (nu sare peste etape)
    trans_p = np.array([
        [0.65, 0.3, 0.05, 0.0],
        [0.2, 0.6, 0.2, 0.0],
        [0.0, 0.25, 0.5, 0.25],
        [0.0, 0.05, 0.3, 0.65]
    ])
    
    # Emisie: Ambiguu la mijloc (2 inghetate)
    emit_p = np.array([
        [0.05, 0.15, 0.80], # FC
        [0.10, 0.60, 0.30], # C (Greedy va alege asta pt 2 ingh)
        [0.30, 0.60, 0.10], # R (SAU asta? Greedy e confuz)
        [0.80, 0.15, 0.05]  # FR
    ])
    return states, obs_vocab, start_p, trans_p, emit_p

# ==========================================
# 3. HELPER FUNCTION PENTRU RULARE VREME
# ==========================================
def run_simulare_vreme(nume_scenariu, params_func):
    print(f"\n--- {nume_scenariu} ---")
    states, obs_vocab, start_p, trans_p, emit_p = params_func()

    while True:
        try:
            n_input = input("Introduceti numarul de zile (ex: 30): ")
            n_zile = int(n_input)
            if n_zile > 0: break
        except ValueError: pass

    # Generare si Rulare
    x_real, y_obs = genereaza_date_generic(n_zile, start_p, trans_p, emit_p)
    x_viterbi = algoritm_viterbi_generic(y_obs, start_p, trans_p, emit_p)
    x_greedy = algoritm_greedy_naiv(y_obs, emit_p)
    
    # Afisare
    print(f"\n{'ZI':<4} {'OBSERVATIE':<10} {'REAL':<10} {'VITERBI':<10} {'GREEDY':<10}")
    print("-" * 60)

    c_vit = 0
    c_grd = 0
    
    for i in range(n_zile):
        if x_real[i] == x_viterbi[i]: c_vit += 1
        if x_real[i] == x_greedy[i]: c_grd += 1
        
        obs_str = obs_vocab[y_obs[i]]
        real = states[x_real[i]]
        vit = states[x_viterbi[i]]
        grd = states[x_greedy[i]]
        
        print(f"{i:<4} {obs_str:<10} {real:<10} {vit:<10} {grd:<10}")

    acc_v = (c_vit / n_zile) * 100
    acc_g = (c_grd / n_zile) * 100
    
    print("-" * 60)
    print(f"Acuratete VITERBI: {acc_v:.2f}%")
    print(f"Acuratete GREEDY:  {acc_g:.2f}%")
    
    return n_zile, acc_v, acc_g

# ==========================================
# 4. SCENARIUL TEXT (PARAMETRI & LOGICA)
# ==========================================
def get_char_index(char):
    if char == ' ': return 26
    return ord(char) - ord('a')

def get_index_char(index):
    if index == 26: return ' '
    return chr(index + ord('a'))

def antreneaza_model_limba(file_path):
    alphabet_size = 27 
    start_counts = np.ones(alphabet_size)
    trans_counts = np.ones((alphabet_size, alphabet_size))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            diacritice = {'ă':'a', 'â':'a', 'î':'i', 'ș':'s', 'ş':'s', 'ț':'t', 'ţ':'t'}
            for k, v in diacritice.items(): text = text.replace(k, v)
            
            text_clean_list = [c if 'a' <= c <= 'z' else ' ' for c in text]
            text_clean = " ".join("".join(text_clean_list).split())
            
            if len(text_clean) > 0:
                first_idx = get_char_index(text_clean[0])
                if 0 <= first_idx < 27: start_counts[first_idx] += 1
                
                for i in range(len(text_clean) - 1):
                    ic = get_char_index(text_clean[i])
                    iin = get_char_index(text_clean[i+1])
                    if 0 <= ic < 27 and 0 <= iin < 27:
                        trans_counts[ic][iin] += 1
                        if text_clean[i] == ' ': start_counts[iin] += 1
    except FileNotFoundError:
        print(f"[!] Fisierul {file_path} lipseste.")

    start_p = start_counts / np.sum(start_counts)
    trans_p = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return start_p, trans_p

def construieste_tastatura_qwerty():
    keyboard_neighbors = {
        'q': 'wa', 'w': 'qase', 'e': 'wsdfr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji', 
        'i': 'ujko', 'o': 'iklp', 'p': 'ol', 'a': 'qwsz', 's': 'qweadzx', 'd': 'wersfxc', 
        'f': 'ertdgcv', 'g': 'rtyfhvb', 'h': 'tyugjbn', 'j': 'yuihknm', 'k': 'uiojlm', 'l': 'iopk', 
        'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
    }
    
    alphabet_size = 27
    emit_p = np.zeros((alphabet_size, alphabet_size))
    
    PROB_CORECT = 0.8
    PROB_VECIN = 0.15
    
    for real_char_code in range(26):
        char = chr(real_char_code + ord('a'))
        neighbors = keyboard_neighbors.get(char, "")
        
        emit_p[real_char_code, :] = 1e-5 
        emit_p[real_char_code, real_char_code] = PROB_CORECT
        
        if neighbors:
            prob = PROB_VECIN / len(neighbors)
            for n in neighbors:
                emit_p[real_char_code, get_char_index(n)] = prob
        
        emit_p[real_char_code] /= emit_p[real_char_code].sum()
                
    emit_p[26, :] = 1e-5
    emit_p[26, 26] = 0.9 
    emit_p[26] /= emit_p[26].sum()
    return emit_p

def genereaza_typo(text_corect, emit_p):
    observatii = []
    text_stricat = ""
    for char in text_corect:
        if not (char.isalpha() or char == ' '): continue
        real_idx = get_char_index(char.lower())
        obs_idx = np.random.choice(range(27), p=emit_p[real_idx])
        observatii.append(obs_idx)
        text_stricat += get_index_char(obs_idx)
    return observatii, text_stricat

def run_corectie_text():
    print("\n--- CORECTIE TEXT (VITERBI vs GREEDY) ---")
    asigura_existenta_training_data()
    print("Antrenare model din 'training_data.txt'...")
    
    start_p, trans_p = antreneaza_model_limba('training_data.txt')
    emit_p = construieste_tastatura_qwerty()
    
    input_text = input("Introduceti un text CORECT (ex: 'ana are mere'): ").lower()
    if not all(c.isalpha() or c == ' ' for c in input_text):
        print("Va rugam folositi doar litere si spatii.")
        return None

    y_obs, text_stricat = genereaza_typo(input_text, emit_p)
    
    print(f"\nOriginal: '{input_text}'")
    print(f"Observat: '{text_stricat}' (Greedy va alege aceste caractere, poate putin corectate)")
    
    # Rulare Viterbi
    path_viterbi = algoritm_viterbi_generic(y_obs, start_p, trans_p, emit_p)
    text_viterbi = "".join([get_index_char(idx) for idx in path_viterbi])
    
    # Rulare Greedy (Practic doar maparea inversa a tastelor fara context)
    path_greedy = algoritm_greedy_naiv(y_obs, emit_p)
    text_greedy = "".join([get_index_char(idx) for idx in path_greedy])
    
    print(f"Viterbi:  '{text_viterbi}'")
    print(f"Greedy:   '{text_greedy}'")
    
    # Calcul Acuratete
    corecte_v = sum(1 for i in range(len(input_text)) if input_text[i] == text_viterbi[i])
    acc_v = (corecte_v / len(input_text)) * 100
    
    corecte_g = sum(1 for i in range(len(input_text)) if input_text[i] == text_greedy[i])
    acc_g = (corecte_g / len(input_text)) * 100
    
    print(f"Acuratete Viterbi: {acc_v:.2f}%")
    print(f"Acuratete Greedy:  {acc_g:.2f}%")
    
    return len(input_text), acc_v, acc_g

# ==========================================
# 5. MENIU PRINCIPAL
# ==========================================
def run():
    # Structura istoric: (ID, Nume_Afisat, Dimensiune, Acc_Viterbi, Acc_Greedy)
    istoric = [] 
    print("=== PROIECT HMM - VITERBI vs GREEDY ===")

    while True:
        print("\nMENIU PRINCIPAL:")
        print("1 - Predictia Vremii STANDARD (2 Stari: Cald/Rece)")
        print("2 - Predictia Vremii COMPLEXA (4 Stari: F.Cald -> F.Rece)")
        print("3 - Corectie Text (Typo Correction)")
        print("0 - Exit")
        
        opt = input("\nSelectati o optiune: ")
        
        if opt == '0': break
        
        elif opt == '1':
            res = run_simulare_vreme("VREME STANDARD (2 STARI)", get_params_vreme_standard)
            if res: istoric.append((1, "Vreme Std (2)", res[0], res[1], res[2]))
            
        elif opt == '2':
            res = run_simulare_vreme("VREME COMPLEXA (4 STARI)", get_params_vreme_complex)
            if res: istoric.append((2, "Vreme Cplx (4)", res[0], res[1], res[2]))
            
        elif opt == '3':
            res = run_corectie_text()
            if res: istoric.append((3, "Corectie Text", res[0], res[1], res[2]))
            
        else:
            print("Optiune invalida.")
        
        input("\n[Enter] pentru a reveni la meniu...")

    # Tabel Final
    if istoric:
        print("\n" + "="*65)
        print(f"{'RAPORT FINAL SESIUNE':^65}")
        print("="*65)
        print(f"{'PROBLEMA':<20} {'DIM':<10} {'ACC. VITERBI':<15} {'ACC. GREEDY':<15}")
        print("-" * 65)
        for rec in istoric:
            print(f"{rec[1]:<20} {rec[2]:<10} {rec[3]:.2f}%{'':<8} {rec[4]:.2f}%")
        print("="*65)

if __name__ == '__main__':
    run()