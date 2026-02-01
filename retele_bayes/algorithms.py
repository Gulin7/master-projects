import numpy as np

def algoritm_viterbi_generic(obs, start_p, trans_p, emit_p):
    """ Viterbi: Gaseste calea cea mai probabila global (contextual). """
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
    """ Greedy: Alege maximul local la fiecare pas. (Folosit doar la Vreme) """
    path = []
    for o in obs:
        best_state_now = np.argmax(emit_p[:, o])
        path.append(best_state_now)
    return path

# Algoritm forward-backward

def algoritm_forward_backward(obs, start_p, trans_p, emit_p):
    """
    Calculeaza starea cea mai probabila pentru fiecare moment t,
    luand in considerare TOATE observatiile (trecute si viitoare).
    Returneaza calea maximizata marginal (MPM).
    """
    n_states = trans_p.shape[0]
    T = len(obs)

    # 1. FORWARD (Alpha)
    # alpha[s, t] = P(obs_0...obs_t, stare_t = s)
    alpha = np.zeros((n_states, T))
    
    # Init (t=0)
    for s in range(n_states):
        alpha[s, 0] = start_p[s] * emit_p[s, obs[0]]
        
    # Recursivitate Forward
    for t in range(1, T):
        for s in range(n_states):
            # Suma peste toate starile anterioare posibile (diferenta fata de Viterbi care lua max)
            sum_probs = 0
            for prev_s in range(n_states):
                sum_probs += alpha[prev_s, t-1] * trans_p[prev_s, s]
            
            alpha[s, t] = sum_probs * emit_p[s, obs[t]]
            
        # Normalizare la fiecare pas pentru a evita underflow la secvențe lungi
        c = np.sum(alpha[:, t])
        if c > 0: alpha[:, t] /= c

    # 2. BACKWARD (Beta)
    # beta[s, t] = P(obs_t+1...obs_T | stare_t = s)
    beta = np.zeros((n_states, T))
    
    # Init (t=T-1) - Probabilitatea e 1 pentru ca nu mai sunt observatii viitoare
    beta[:, T-1] = 1.0
    
    # Recursivitate Backward (mergem de la T-2 pana la 0)
    for t in range(T-2, -1, -1):
        for s in range(n_states):
            sum_probs = 0
            for next_s in range(n_states):
                # Tranzitie s->next_s * Emisie in next_s * Beta din next_s
                sum_probs += trans_p[s, next_s] * emit_p[next_s, obs[t+1]] * beta[next_s, t+1]
            beta[s, t] = sum_probs
            
        # Optional: Normalizare
        # c = np.sum(beta[:, t])
        # if c > 0: beta[:, t] /= c

    # 3. COMBINARE (Gamma / Posterior)
    # gamma[s, t] = P(stare_t = s | TOATE observatiile)
    posterior = np.zeros((n_states, T))
    
    # path_mpm va contine starea cu probabilitatea maxima la fiecare pas
    path_mpm = []
    
    for t in range(T):
        # gamma ~ alpha * beta
        raw_probs = alpha[:, t] * beta[:, t]
        
        norm_factor = np.sum(raw_probs)
        if norm_factor == 0:
            norm_factor = 1.0 # Evitam impartirea la 0
            
        posterior[:, t] = raw_probs / norm_factor
        
        # Alegem starea cu probabilitatea maxima (MPM)
        best_state = np.argmax(posterior[:, t])
        path_mpm.append(best_state)
        
    return path_mpm

# Algoritm CRF

def algoritm_crf_inference(obs, start_p, trans_p, emit_p):
    """
    Simuleaza inferenta unui CRF (Conditional Random Field).
    Diferenta fata de Viterbi clasic:
    1. Transforma probabilitatile in 'Weights' (Scoruri) folosind Logaritm.
    2. Foloseste ADUNARE in loc de INMULTIRE (Max-Sum Algorithm).
    """
    n_states = trans_p.shape[0]
    T = len(obs)
    
    # Pasul 1: Conversia Probabilitati -> Ponderi (Log-Space)
    # Adaugam un epsilon (1e-10) ca sa nu facem log(0) care da -infinit
    eps = 1e-10
    w_start = np.log(start_p + eps)
    w_trans = np.log(trans_p + eps)
    w_emit  = np.log(emit_p + eps)
    
    # Pasul 2: Initializare (Score Table)
    score_table = np.zeros((n_states, T))
    backpointer = np.zeros((n_states, T), dtype=int)
    
    # t=0: Score = Start_Weight + Emission_Weight
    for s in range(n_states):
        score_table[s, 0] = w_start[s] + w_emit[s, obs[0]]
        
    # Pasul 3: Recursivitate (Max-Sum)
    for t in range(1, T):
        for s in range(n_states):
            # Calculam scorul pentru tranzitia din fiecare stare anterioara
            # FORMULA CRF: Score_Prev + Weight_Tranzitie + Weight_Emisie
            candidate_scores = [score_table[s_prev, t-1] + w_trans[s_prev, s] + w_emit[s, obs[t]] 
                                for s_prev in range(n_states)]
            
            # Alegem MAXIMUL (nu suma, nu produsul)
            score_table[s, t] = np.max(candidate_scores)
            backpointer[s, t] = np.argmax(candidate_scores)
            
    # Pasul 4: Backtracking (Identic cu Viterbi)
    best_last_state = np.argmax(score_table[:, T-1])
    best_path = [best_last_state]
    
    for t in range(T-1, 0, -1):
        prev_state = backpointer[best_path[-1], t]
        best_path.append(prev_state)
        
    best_path.reverse()
    return best_path

# ==============================================================================
# LEVENSHTEIN PONDERAT (KEYBOARD AWARE)
# ==============================================================================

# Definim vecinii in afara functiei pentru performanta
KEYBOARD_ADJACENCY = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdfr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji', 
    'i': 'ujko', 'o': 'iklp', 'p': 'ol', 'a': 'qwsz', 's': 'qweadzx', 'd': 'wersfxc', 
    'f': 'ertdgcv', 'g': 'rtyfhvb', 'h': 'tyugjbn', 'j': 'yuihknm', 'k': 'uiojlm', 'l': 'iopk', 
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

def get_substitution_cost(char1, char2):
    """
    Returneaza costul inlocuirii lui char1 cu char2.
    - 0 daca sunt identice
    - 0.5 daca sunt vecini pe tastatura (greseala probabila)
    - 1.0 altfel
    """
    if char1 == char2:
        return 0
    
    # Verificam daca sunt vecini
    # char1 in neighbors of char2 OR char2 in neighbors of char1
    neighbors1 = KEYBOARD_ADJACENCY.get(char1, "")
    if char2 in neighbors1:
        return 0.5 # Penalizare mica pentru 'fat finger'
        
    return 1.0 # Penalizare standard

def compute_levenshtein_distance(s1, s2):
    """
    Calculeaza distanta Levenshtein PONDERATA.
    Tine cont de layout-ul tastaturii QWERTY.
    """
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Initializare rand anterior (costuri de stergere pure = 1 per litera)
    previous_row = [i * 1.0 for i in range(len(s2) + 1)]
    
    for i, c1 in enumerate(s1):
        # Primul element din randul curent este costul insertiilor (i+1)
        current_row = [(i + 1) * 1.0]
        
        for j, c2 in enumerate(s2):
            # Costuri operatii
            insertion = previous_row[j + 1] + 1.0
            deletion = current_row[j] + 1.0
            
            # AICI ESTE SCHIMBAREA CHEIE: Cost variabil de substitutie
            subst_cost = get_substitution_cost(c1, c2)
            substitution = previous_row[j] + subst_cost
            
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    
    return previous_row[-1]

def corrector_levenshtein(input_sentence, vocabular):
    cuvinte_input = input_sentence.split()
    cuvinte_corectate = []
    
    for cuvant_gresit in cuvinte_input:
        best_word = cuvant_gresit
        min_distance = float('inf')
        
        for candidat in vocabular:
            # Optimizare: Daca diferenta de lungime e mai mare decat min_distance gasit deja,
            # nu are sens sa calculam (distanta e cel putin diferenta de lungime).
            if abs(len(cuvant_gresit) - len(candidat)) > min_distance:
                continue

            dist = compute_levenshtein_distance(cuvant_gresit, candidat)
            
            if dist < min_distance:
                min_distance = dist
                best_word = candidat
            
            # Daca gasim potrivire perfecta (0.0), ne oprim
            if dist == 0.0:
                break
        
        cuvinte_corectate.append(best_word)
        
    return " ".join(cuvinte_corectate)

# Jaro-Winkler Similarity
def jaro_winkler_similarity(s1, s2):
    """
    Calculeaza similaritatea Jaro-Winkler (0.0 la 1.0).
    1.0 inseamna identic. Favorizeaza potrivirile la inceputul cuvantului.
    """
    # Daca sunt identice
    if s1 == s2: return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0: return 0.0

    # Fereastra de potrivire (cat de departe cautam un caracter)
    match_distance = max(0, (max(len1, len2) // 2) - 1)

    # 1. Calculam Matches (m)
    matches = 0
    hash_s1 = [0] * len1
    hash_s2 = [0] * len2

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2[j] == s1[i] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                matches += 1
                break

    if matches == 0: return 0.0

    # 2. Calculam Transpositions (t)
    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
    t /= 2

    # 3. Jaro Distance
    # m/|s1| + m/|s2| + (m-t)/m
    jaro_dist = ((matches / len1) + (matches / len2) + ((matches - t) / matches)) / 3.0

    # 4. Winkler Bonus (Prefix scale)
    # Bonus pentru caractere comune la inceput (max 4)
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]: prefix += 1
        else: break
        
    # Scaling factor standard = 0.1
    jaro_winkler = jaro_dist + (prefix * 0.1 * (1 - jaro_dist))
    
    return jaro_winkler

def corrector_jaro_winkler(input_sentence, vocabular):
    """
    Corecteaza propozitia folosind Jaro-Winkler.
    Atentie: Aici cautam scorul MAXIM (cel mai similar), nu distanta minima.
    """
    cuvinte_input = input_sentence.split()
    cuvinte_corectate = []
    
    for cuvant_gresit in cuvinte_input:
        best_word = cuvant_gresit
        max_score = -1.0
        
        for candidat in vocabular:
            score = jaro_winkler_similarity(cuvant_gresit, candidat)
            
            if score > max_score:
                max_score = score
                best_word = candidat
            
            # Daca gasim match perfect
            if score == 1.0:
                break
        
        cuvinte_corectate.append(best_word)
        
    return " ".join(cuvinte_corectate)