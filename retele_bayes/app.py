import streamlit as st
import pandas as pd
import numpy as np
import time

# Import your existing modules
import utils
import sequence_algorithms
import text_correction
import weather_model
import text_model
import pos_tagging

# ==============================================================================
# CONFIGURATION & CACHING
# ==============================================================================
st.set_page_config(page_title="Viterbi Algorithm Project", layout="wide")

# Use @st.cache_resource to load heavy models ONLY ONCE
@st.cache_resource
def get_spacy_model():
    return pos_tagging.load_spacy_model()

@st.cache_resource
def get_hmm_model():
    return pos_tagging.load_pos_model()

# ==============================================================================
# SIDEBAR MENU
# ==============================================================================
st.sidebar.title("Viterbi Project")
menu = st.sidebar.radio(
    "Select Module:",
    ["Weather Prediction", "Text Correction", "POS Tagging", "About"]
)

# ==============================================================================
# 1. WEATHER MODULE
# ==============================================================================
if menu == "Weather Prediction":
    st.header("🌦️ Weather Prediction Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        scenario = st.selectbox("Scenario", ["Standard (2 States)", "Complex (4 States)"])
        days = st.slider("Number of Days", min_value=10, max_value=500, value=30)
        
        if st.button("Run Simulation"):
            # Load params based on selection
            if scenario == "Standard (2 States)":
                states, obs_vocab, start_p, trans_p, emit_p = weather_model.get_params_weather_standard()
            else:
                states, obs_vocab, start_p, trans_p, emit_p = weather_model.get_params_weather_complex()
            
            # 1. Generate Data
            x_real, y_obs = utils.generate_generic_data(days, start_p, trans_p, emit_p)
            
            # 2. Run Algorithms
            x_viterbi = sequence_algorithms.generic_viterbi_algorithm(y_obs, start_p, trans_p, emit_p)
            x_fb = sequence_algorithms.forward_backward_algorithm(y_obs, start_p, trans_p, emit_p)
            x_crf = sequence_algorithms.crf_inference_algorithm(y_obs, start_p, trans_p, emit_p)
            x_greedy = sequence_algorithms.naive_greedy_algorithm(y_obs, emit_p)
            
            # 3. Calculate Accuracy
            acc_v = np.mean(np.array(x_real) == np.array(x_viterbi)) * 100
            acc_fb = np.mean(np.array(x_real) == np.array(x_fb)) * 100
            acc_crf = np.mean(np.array(x_real) == np.array(x_crf)) * 100
            acc_g = np.mean(np.array(x_real) == np.array(x_greedy)) * 100
            
            # Store results in Session State to persist them
            st.session_state['weather_results'] = {
                'data': zip(range(days), 
                            [obs_vocab[y] for y in y_obs], 
                            [states[x] for x in x_real],
                            [states[x] for x in x_viterbi],
                            [states[x] for x in x_fb],
                            [states[x] for x in x_crf],
                            [states[x] for x in x_greedy]),
                'metrics': (acc_v, acc_fb, acc_crf, acc_g)
            }

    # Display Results if they exist
    if 'weather_results' in st.session_state:
        res = st.session_state['weather_results']
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Viterbi", f"{res['metrics'][0]:.1f}%")
        m2.metric("Forward-Backward", f"{res['metrics'][1]:.1f}%")
        m3.metric("CRF (Inference)", f"{res['metrics'][2]:.1f}%")
        m4.metric("Greedy", f"{res['metrics'][3]:.1f}%")
        
        # Data Table
        df = pd.DataFrame(res['data'], columns=['Day', 'Observation', 'Real', 'Viterbi', 'Fwd-Bwd', 'CRF', 'Greedy'])
        st.dataframe(df, use_container_width=True)

# ==============================================================================
# 2. TEXT CORRECTION MODULE (ROMANIAN DEFAULT + METRICS)
# ==============================================================================
elif menu == "Text Correction":
    st.header("✍️ Comparatie Corectie Text")
    
    # Initialize files if needed
    utils.ensure_file_exists('training_data.txt')
    utils.ensure_file_exists('training_words.txt')
    
    # --- HELPER FUNCTIONS FOR ACCURACY ---
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
    # -------------------------------------

    col1, col2 = st.columns([1, 1])
    
    with col1:
        # INPUT DEFAULT IN ROMANA
        input_text = st.text_input("Introdu Text Corect (ex: 'ana are mere'):", "ana are mere")
    
    if st.button("Genereaza Typo & Corecteaza"):
        # Load Models
        start_p, trans_p = text_model.train_language_model('training_data.txt')
        emit_p = text_model.build_qwerty_keyboard()
        vocab = text_model.extract_vocabulary('training_words.txt')
        
        # Generate Error (Lower case for fairness)
        clean_input = input_text.lower()
        y_obs, broken_text = text_model.generate_typo(clean_input, emit_p)
        
        st.subheader(f"Observat (Typo): :red[{broken_text}]")
        
        # Run Algorithms
        # A. Viterbi
        path_v = sequence_algorithms.generic_viterbi_algorithm(y_obs, start_p, trans_p, emit_p)
        txt_v = "".join([text_model.get_index_char(idx) for idx in path_v])
        
        # B. Levenshtein
        txt_lev = text_correction.levenshtein_corrector(broken_text, vocab)
        
        # C. Jaro-Winkler
        txt_jaro = text_correction.jaro_winkler_corrector(broken_text, vocab)
        
        # CALCUL METRICI
        # Viterbi Stats
        acc_v_char = calc_acc(clean_input, txt_v)
        acc_v_word = calc_word_acc(clean_input, txt_v)
        
        # Levenshtein Stats
        acc_lev_char = calc_acc(clean_input, txt_lev)
        acc_lev_word = calc_word_acc(clean_input, txt_lev)
        
        # Jaro Stats
        acc_jaro_char = calc_acc(clean_input, txt_jaro)
        acc_jaro_word = calc_word_acc(clean_input, txt_jaro)
        
        # Display Results in Cards
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.info(f"**Viterbi (Context)**\n\nRESULT: `{txt_v}`\n\n"
                    f"🎯 **Char:** {acc_v_char:.1f}%\n\n"
                    f"📖 **Word:** {acc_v_word:.1f}%")
            
        with c2:
            st.success(f"**Levenshtein (Dict)**\n\nRESULT: `{txt_lev}`\n\n"
                       f"🎯 **Char:** {acc_lev_char:.1f}%\n\n"
                       f"📖 **Word:** {acc_lev_word:.1f}%")
            
        with c3:
            st.warning(f"**Jaro-Winkler (Sim)**\n\nRESULT: `{txt_jaro}`\n\n"
                       f"🎯 **Char:** {acc_jaro_char:.1f}%\n\n"
                       f"📖 **Word:** {acc_jaro_word:.1f}%")

# ==============================================================================
# 3. POS TAGGING MODULE
# ==============================================================================
elif menu == "POS Tagging":
    st.header("🏷️ Part-of-Speech Tagging")
    st.write("Comparing **HMM Viterbi** (Brown Corpus) vs **spaCy** (Neural Network).")
    
    user_sent = st.text_input("Enter English Sentence:", "I need to make this work perfectly .")
    
    if st.button("Analyze Syntax"):
        # 1. Load Models
        hmm_model = get_hmm_model()
        spacy_nlp = get_spacy_model() # ensures spacy is loaded
        
        tags = hmm_model['tags']
        word2idx = hmm_model['word2idx']
        
        words = user_sent.split()
        
        # 2. Run Viterbi (With NOUN Defaulting logic)
        noun_tag_idx = tags.index('NOUN') if 'NOUN' in tags else 0
        
        # Logic: If word in dict, use index. Else, use NOUN index.
        obs_indices = [word2idx.get(w.lower(), noun_tag_idx) for w in words]
        
        path_v = sequence_algorithms.generic_viterbi_algorithm(
            obs_indices, 
            hmm_model['start_p'], 
            hmm_model['trans_p'], 
            hmm_model['emit_p']
        )
        viterbi_tags = [tags[idx] for idx in path_v]
        
        # 3. Run SpaCy
        spacy_tags = pos_tagging.get_spacy_tags(words)
        
        # 4. Build Comparison Data
        data = []
        matches = 0
        for i, w in enumerate(words):
            is_match = viterbi_tags[i] == spacy_tags[i]
            if is_match: matches += 1
            
            # Add status icon
            status = "✅" if is_match else "❌"
            data.append([w, viterbi_tags[i], spacy_tags[i], status])
            
        # Display Table
        df = pd.DataFrame(data, columns=["Word", "Viterbi Tag", "SpaCy Tag", "Match"])
        
        # Styled Table
        st.dataframe(
            df.style.apply(lambda x: ['background-color: #ffcccc' if x['Match'] == "❌" else '' for i in x], axis=1),
            use_container_width=True
        )
        
        # Accuracy Score
        accuracy = (matches / len(words)) * 100
        st.metric("Agreement Score", f"{accuracy:.1f}%")

# ==============================================================================
# ABOUT
# ==============================================================================
elif menu == "About":
    st.subheader("About this Project")
    st.markdown("""
    This application demonstrates the **Viterbi Algorithm** applied to three distinct domains:
    
    1.  **Temporal Sequences:** Predicting hidden weather states based on observations (Ice cream consumption).
    2.  **Text Correction:** Reconstructing intended text from "fat-finger" typos on a QWERTY keyboard.
    3.  **NLP Tagging:** Assigning grammatical parts of speech to words.
    
    **Stack:** Python, NumPy, Streamlit, SpaCy, NLTK.
    """)