# ==============================================================================
# WEIGHTED LEVENSHTEIN (KEYBOARD AWARE)
# ==============================================================================

KEYBOARD_ADJACENCY = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdfr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji', 
    'i': 'ujko', 'o': 'iklp', 'p': 'ol', 'a': 'qwsz', 's': 'qweadzx', 'd': 'wersfxc', 
    'f': 'ertdgcv', 'g': 'rtyfhvb', 'h': 'tyugjbn', 'j': 'yuihknm', 'k': 'uiojlm', 'l': 'iopk', 
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

def get_substitution_cost(char1, char2):
    if char1 == char2:
        return 0
    neighbors1 = KEYBOARD_ADJACENCY.get(char1, "")
    if char2 in neighbors1:
        return 0.5 
    return 1.0

def compute_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = [i * 1.0 for i in range(len(s2) + 1)]
    
    for i, c1 in enumerate(s1):
        current_row = [(i + 1) * 1.0]
        for j, c2 in enumerate(s2):
            insertion = previous_row[j + 1] + 1.0
            deletion = current_row[j] + 1.0
            subst_cost = get_substitution_cost(c1, c2)
            substitution = previous_row[j] + subst_cost
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein_corrector(input_sentence, vocabulary):
    input_words = input_sentence.split()
    corrected_words = []
    
    for wrong_word in input_words:
        best_word = wrong_word
        min_distance = float('inf')
        
        for candidate in vocabulary:
            if abs(len(wrong_word) - len(candidate)) > min_distance:
                continue
            dist = compute_levenshtein_distance(wrong_word, candidate)
            if dist < min_distance:
                min_distance = dist
                best_word = candidate
            if dist == 0.0:
                break
        
        corrected_words.append(best_word)
        
    return " ".join(corrected_words)

# ==============================================================================
# JARO-WINKLER
# ==============================================================================

def jaro_winkler_similarity(s1, s2):
    if s1 == s2: return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0: return 0.0

    match_distance = max(0, (max(len1, len2) // 2) - 1)
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

    jaro_dist = ((matches / len1) + (matches / len2) + ((matches - t) / matches)) / 3.0

    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]: prefix += 1
        else: break
        
    jaro_winkler = jaro_dist + (prefix * 0.1 * (1 - jaro_dist))
    return jaro_winkler

def jaro_winkler_corrector(input_sentence, vocabulary):
    input_words = input_sentence.split()
    corrected_words = []
    
    for wrong_word in input_words:
        best_word = wrong_word
        max_score = -1.0
        
        for candidate in vocabulary:
            score = jaro_winkler_similarity(wrong_word, candidate)
            if score > max_score:
                max_score = score
                best_word = candidate
            if score == 1.0:
                break
        
        corrected_words.append(best_word)
        
    return " ".join(corrected_words)