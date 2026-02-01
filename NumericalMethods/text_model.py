import numpy as np

def get_char_index(char):
    if char == ' ': return 26
    return ord(char) - ord('a')

def get_index_char(index):
    if index == 26: return ' '
    return chr(index + ord('a'))

def train_language_model(file_path):
    alphabet_size = 27 
    start_counts = np.ones(alphabet_size)
    trans_counts = np.ones((alphabet_size, alphabet_size))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            diacritics = {'ă':'a', 'â':'a', 'î':'i', 'ș':'s', 'ş':'s', 'ț':'t', 'ţ':'t'}
            for k, v in diacritics.items(): text = text.replace(k, v)
            
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
        print(f"[TextModel] File {file_path} is missing.")

    start_p = start_counts / np.sum(start_counts)
    trans_p = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return start_p, trans_p

def build_qwerty_keyboard():
    keyboard_neighbors = {
        'q': 'wa', 'w': 'qase', 'e': 'wsdfr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji', 
        'i': 'ujko', 'o': 'iklp', 'p': 'ol', 'a': 'qwsz', 's': 'qweadzx', 'd': 'wersfxc', 
        'f': 'ertdgcv', 'g': 'rtyfhvb', 'h': 'tyugjbn', 'j': 'yuihknm', 'k': 'uiojlm', 'l': 'iopk', 
        'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
    }
    
    alphabet_size = 27
    emit_p = np.zeros((alphabet_size, alphabet_size))
    
    # --- PROBABILITY CONFIGURATION ---
    # Sum should be 1.0 (or close, we normalize at the end)
    PROB_CORRECT = 0.70   # Correct key
    PROB_NEIGHBOR = 0.20  # Neighbor key (fat finger)
    PROB_RANDOM = 0.10    # Any other key (pure noise)
    
    for real_char_code in range(26):
        char = chr(real_char_code + ord('a'))
        neighbors = keyboard_neighbors.get(char, "")
        
        # 1. Distribute "Random Noise" equally across the ENTIRE row
        # So no key has probability 0 (avoids Viterbi errors)
        emit_p[real_char_code, :] = PROB_RANDOM / alphabet_size
        
        # 2. Add CORRECT probability to the specific index
        # (Use += to add on top of background noise)
        emit_p[real_char_code, real_char_code] += PROB_CORRECT
        
        # 3. Add NEIGHBOR probabilities
        if neighbors:
            prob_per_neighbor = PROB_NEIGHBOR / len(neighbors)
            for n in neighbors:
                emit_p[real_char_code, get_char_index(n)] += prob_per_neighbor
        
        # 4. MANDATORY NORMALIZATION (Row sum = 1.0)
        emit_p[real_char_code] /= emit_p[real_char_code].sum()
                
    # Special handling for SPACE (index 26)
    # Space is harder to miss, but we leave a little noise
    emit_p[26, :] = 1e-5 
    emit_p[26, 26] = 0.95
    emit_p[26] /= emit_p[26].sum()
    
    return emit_p

def generate_typo(correct_text, emit_p):
    observations = []
    broken_text = ""
    for char in correct_text:
        if not (char.isalpha() or char == ' '): continue
        real_idx = get_char_index(char.lower())
        obs_idx = np.random.choice(range(27), p=emit_p[real_idx])
        observations.append(obs_idx)
        broken_text += get_index_char(obs_idx)
    return observations, broken_text

def extract_vocabulary(file_path):
    """
    Reads training text and returns a list of unique words (Dictionary).
    Used for Levenshtein.
    """
    vocabulary = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            
            # Clean diacritics (to be consistent with the rest of the project)
            diacritics = {'ă':'a', 'â':'a', 'î':'i', 'ș':'s', 'ş':'s', 'ț':'t', 'ţ':'t'}
            for k, v in diacritics.items(): text = text.replace(k, v)
            
            # Replace anything not a letter with space, then split
            text_clean = "".join([c if 'a' <= c <= 'z' else ' ' for c in text])
            words = text_clean.split()
            
            # Add to set (automatically removes duplicates)
            vocabulary.update(words)
            
    except FileNotFoundError:
        print(f"[TextModel] File {file_path} is missing.")
        return ["ana", "are", "mere"] # Fallback
        
    return list(vocabulary)