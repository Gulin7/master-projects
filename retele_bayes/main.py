import utils
import sequence_algorithms
import text_correction
import weather_model
import text_model
import pos_tagging

# ==============================================================================
# WEATHER HANDLER (Runs ALL algorithms)
# ==============================================================================
def run_weather_simulation(scenario_name, params_func):
    print(f"\n--- {scenario_name} ---")
    states, obs_vocab, start_p, trans_p, emit_p = params_func()

    while True:
        try:
            n_input = input("Enter number of days (e.g., 30): ")
            n_days = int(n_input)
            if n_days > 0: break
        except ValueError: pass

    # 1. Generate Data
    x_real, y_obs = utils.generate_generic_data(n_days, start_p, trans_p, emit_p)
    
    # 2. Run Algorithms
    x_viterbi = sequence_algorithms.generic_viterbi_algorithm(y_obs, start_p, trans_p, emit_p)
    x_fb = sequence_algorithms.forward_backward_algorithm(y_obs, start_p, trans_p, emit_p)
    x_crf = sequence_algorithms.crf_inference_algorithm(y_obs, start_p, trans_p, emit_p)
    x_greedy = sequence_algorithms.naive_greedy_algorithm(y_obs, emit_p)
    
    # 3. Display Table
    print(f"\n{'DAY':<4} {'OBS':<12} {'REAL':<10} {'VITERBI':<10} {'FW-BW':<10} {'CRF':<10} {'GREEDY':<10}")
    print("-" * 40)

    c_vit = c_fb = c_crf = c_grd = 0
    
    for i in range(n_days):
        if x_real[i] == x_viterbi[i]: c_vit += 1
        if x_real[i] == x_fb[i]: c_fb += 1
        if x_real[i] == x_crf[i]: c_crf += 1
        if x_real[i] == x_greedy[i]: c_grd += 1
        
        obs_str = obs_vocab[y_obs[i]]
        s_real = states[x_real[i]][:9]
        s_vit  = states[x_viterbi[i]][:9]
        s_fb   = states[x_fb[i]][:9]
        s_crf  = states[x_crf[i]][:9]
        s_grd  = states[x_greedy[i]][:9]
        
        print(f"{i:<4} {obs_str:<12} {s_real:<10} {s_vit:<10} {s_fb:<10} {s_crf:<10} {s_grd:<10}")

    acc_v   = (c_vit / n_days) * 100
    acc_fb  = (c_fb / n_days) * 100
    acc_crf = (c_crf / n_days) * 100
    acc_g   = (c_grd / n_days) * 100
    
    print("-" * 40)
    print(f"Accuracy VITERBI: {acc_v:.2f}%")
    print(f"Accuracy FW-BW:   {acc_fb:.2f}%")
    print(f"Accuracy CRF:     {acc_crf:.2f}%")
    print(f"Accuracy GREEDY:  {acc_g:.2f}%")
    
    return n_days, acc_v, acc_fb, acc_crf, acc_g

# ==============================================================================
# TEXT HANDLER (Viterbi vs Levenshtein vs Jaro-Winkler)
# ==============================================================================
def run_text_correction():
    print("\n--- TEXT CORRECTION (3 ALGO COMPARISON) ---")
    
    # Ensure files exist
    utils.ensure_file_exists('training_data.txt')  # For Context (Viterbi)
    utils.ensure_file_exists('training_words.txt') # For Dictionary (Levenshtein/Jaro)

    print("Loading Models...")
    
    # 1. HMM / Viterbi
    start_p, trans_p = text_model.train_language_model('training_data.txt')
    emit_p = text_model.build_qwerty_keyboard()
    
    # 2. Vocabulary (Levenshtein & Jaro-Winkler)
    vocabulary = text_model.extract_vocabulary('training_words.txt')
    print(f"Vocabulary loaded: {len(vocabulary)} unique words.")
    
    input_text = input("Enter CORRECT text (e.g. 'ana are mere'): ").lower()
    if not all(c.isalpha() or c == ' ' for c in input_text):
        print("Please use only letters and spaces.")
        return None

    y_obs, broken_text = text_model.generate_typo(input_text, emit_p)
    
    print(f"\nOriginal: '{input_text}'")
    print(f"Observed: '{broken_text}'")
    
    # A. Viterbi (Context)
    path_v = sequence_algorithms.generic_viterbi_algorithm(y_obs, start_p, trans_p, emit_p)
    txt_v = "".join([text_model.get_index_char(idx) for idx in path_v])
    
    # B. Levenshtein (Dictionary - Min Distance)
    txt_lev = text_correction.levenshtein_corrector(broken_text, vocabulary)
    
    # C. Jaro-Winkler (Dictionary - Max Similarity)
    txt_jaro = text_correction.jaro_winkler_corrector(broken_text, vocabulary)
    
    print("-" * 40)
    print(f"Viterbi (HMM):      '{txt_v}'")
    print(f"Levenshtein (ED):   '{txt_lev}'")
    print(f"Jaro-Winkler (Sim): '{txt_jaro}'")
    print("-" * 40)
    
    # Helper function for accuracy
    def calc_acc(real, pred):
        min_len = min(len(real), len(pred))
        correct = sum(1 for i in range(min_len) if real[i] == pred[i])
        diff = abs(len(real) - len(pred))
        return (correct / (len(real) + diff)) * 100
    
    def calc_word_acc(real_text, pred_text):
        real_words = real_text.split()
        pred_words = pred_text.split()
        matches = 0
        min_len = min(len(real_words), len(pred_words))
        
        for i in range(min_len):
            if real_words[i] == pred_words[i]:
                matches += 1
        if len(real_words) == 0: return 0.0
        return (matches / len(real_words)) * 100

    acc_v = calc_acc(input_text, txt_v)
    acc_lev = calc_acc(input_text, txt_lev)
    acc_jaro = calc_acc(input_text, txt_jaro)
    
    acc_v_word = calc_word_acc(input_text, txt_v)
    acc_lev_word = calc_word_acc(input_text, txt_lev)
    acc_jaro_word = calc_word_acc(input_text, txt_jaro)

    print("-" * 40)
    print(f"METRIC         | {'VITERBI':<10} | {'LEVENSHTEIN':<12} | {'JARO-W':<10}")
    print("-" * 40)
    print(f"Characters (%) | {acc_v:<10.2f} | {acc_lev:<12.2f} | {acc_jaro:<10.2f}")
    print(f"Words      (%) | {acc_v_word:<10.2f} | {acc_lev_word:<12.2f} | {acc_jaro_word:<10.2f}")
    print("-" * 40)

    return len(input_text), max(acc_v, acc_v_word), max(acc_lev, acc_lev_word), max(acc_jaro, acc_jaro_word)

# ==============================================================================
# POS TAGGING HANDLER (Viterbi vs SpaCy)
# ==============================================================================

POS_MODEL = None

def run_pos_tagging_inference():
    global POS_MODEL
    # 1. Load your Viterbi Model
    if POS_MODEL is None:
        POS_MODEL = pos_tagging.load_pos_model()
    
    tags = POS_MODEL['tags']
    word2idx = POS_MODEL['word2idx']
    
    text = input("\nEnter English sentence: ")
    words = text.split() 
    
    # --- A. RUN VITERBI ---
    # Handle unknown words by defaulting to 0
    # Find the index of NOUN in your tag list
    # We use tags.index('NOUN') because 'tags' is a list of strings
    noun_tag_idx = tags.index('NOUN') if 'NOUN' in tags else 0
    obs_indices = [word2idx.get(w.lower(), noun_tag_idx) for w in words]
    
    path_v = sequence_algorithms.generic_viterbi_algorithm(
        obs_indices, 
        POS_MODEL['start_p'], 
        POS_MODEL['trans_p'], 
        POS_MODEL['emit_p']
    )
    viterbi_tags = [tags[idx] for idx in path_v]
    
    # --- B. RUN SPACY (ALIGNED) ---
    spacy_tags = pos_tagging.get_spacy_tags(words)
    
    # --- DISPLAY TABLE ---
    print(f"\n{'WORD':<15} | {'VITERBI':<12} | {'SPACY':<12}")
    print("-" * 40)
    
    correct_matches = 0
    
    for i in range(len(words)):
        diff_marker = " "
        if viterbi_tags[i] == spacy_tags[i]:
            correct_matches += 1
        else:
            diff_marker = "(*)"
            
        print(f"{words[i]:<15} | {viterbi_tags[i]:<12} | {spacy_tags[i]:<12} {diff_marker}")
        
    print("-" * 40)

# ==============================================================================
# MAIN MENU
# ==============================================================================
def run():
    weather_history = [] 
    text_history = []

    print("=== VITERBI ALGORITHM PROJECT ===")

    while True:
        print("\nMENU:")
        print("1 - Weather Prediction (Standard)")
        print("2 - Weather Prediction (Complex)")
        print("3 - Text Correction")
        print("4 - Run POS Tagging")
        print("5 - Re-train POS Tagging Model")
        print("0 - Exit")
        
        opt = input("\nSelect: ")
        
        if opt == '0': break
        elif opt == '1':
            res = run_weather_simulation("STANDARD WEATHER", weather_model.get_params_weather_standard)
            if res: weather_history.append(("Std Weather", res[0], res[1], res[2], res[3], res[4]))
        elif opt == '2':
            res = run_weather_simulation("COMPLEX WEATHER", weather_model.get_params_weather_complex)
            if res: weather_history.append(("Cplx Weather", res[0], res[1], res[2], res[3], res[4]))
        elif opt == '3':
            res = run_text_correction()
            if res: text_history.append(("Text", res[0], res[1], res[2], res[3]))
        elif opt == '4':
            run_pos_tagging_inference()
        elif opt == '5':
            pos_tagging.train_and_save_model()
            global POS_MODEL
            POS_MODEL = None
        
        if opt != '0': input("\n[Enter]...")

    print("\n" + "-" * 40)
    print(f"{'FINAL SESSION REPORT':^40}")
    print("-" * 40)
    
    # TABLE 1: WEATHER
    if weather_history:
        print("\n--- WEATHER PREDICTION RESULTS ---")
        print(f"{'SCENARIO':<15} {'LEN':<5} {'VITERBI':<8} {'FW-BW':<8} {'CRF':<8} {'GREEDY':<8}")
        print("-" * 40)
        for rec in weather_history:
            print(f"{rec[0]:<15} {rec[1]:<5} {rec[2]:.1f}%{'':<3} {rec[3]:.1f}%{'':<3} {rec[4]:.1f}%{'':<3} {rec[5]:.1f}%")
            
    # TABLE 2: TEXT
    if text_history:
        print("\n--- TEXT CORRECTION RESULTS ---")
        print(f"{'SCENARIO':<15} {'Length':<10} {'VITERBI':<10} {'LEVENSHTEIN':<12} {'JARO-W':<10}")
        print("-" * 40)
        for rec in text_history:
            print(f"{rec[0]:<15} {rec[1]:<10} {rec[2]:.2f}%{'':<5} {rec[3]:.2f}%{'':<6} {rec[4]:.2f}%")
            
    print("-" * 40)

if __name__ == '__main__':
    run()