import numpy as np
import nltk
import pickle
import os
from nltk.corpus import brown
import sequence_algorithms

import spacy
from spacy.tokens import Doc

# Global variable to cache the loaded model
SPACY_NLP = None

def load_spacy_model():
    """
    Loads the pre-trained spaCy model.
    Acts as the 'training' phase since it loads heavy neural network weights.
    """
    global SPACY_NLP
    if SPACY_NLP is None:
        print("Loading spaCy model (en_core_web_sm)...")
        try:
            # Disable 'parser' and 'ner' to speed it up (we only need 'tagger')
            SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("Error: Model 'en_core_web_sm' not found.")
            print("Please run: python -m spacy download en_core_web_sm")
            return None
    return SPACY_NLP

def get_spacy_tags(word_list):
    """
    Input: List of words (e.g., ["I", "like", "pizza"])
    Output: List of POS tags corresponding to the words.
    """
    nlp = load_spacy_model()
    if nlp is None: 
        return ["ERR"] * len(word_list)

    doc = Doc(nlp.vocab, words=word_list)
    
    for name, pipe in nlp.pipeline:
        if name in ['tok2vec', 'tagger', 'attribute_ruler']:
            doc = pipe(doc)
    
    # Return the Universal POS tags
    return [token.pos_ for token in doc]

MODEL_PATH = "pos_hmm_model.pkl"

def train_and_save_model():
    """Trains on 75% of Brown Corpus, tests on 25%, and saves parameters."""
    print("Loading Brown Corpus...")
    tagged_sents_view = brown.tagged_sents(tagset='universal')
    
    # --- NORMALIZATION STEP ---
    # We map NLTK/Brown tags to match spaCy's Universal Dependencies tags
    print("Normalizing tags to match spaCy ('.'->'PUNCT', 'PRT'->'PART', etc)...")
    
    tag_mapping = {
        '.': 'PUNCT',
        'PRT': 'PART',
        'CONJ': 'CCONJ'
    }
    
    tagged_sents = []
    for sent in tagged_sents_view:
        new_sent = []
        for word, tag in sent:
            tag_str = str(tag) 
            word_str = str(word)
            
            # Apply mapping if exists, otherwise keep original tag
            norm_tag = tag_mapping.get(tag_str, tag_str)
            new_sent.append((word_str, norm_tag))
        tagged_sents.append(new_sent)
    # --------------------------
    
    # 1. 75% / 25% Split
    split_idx = int(len(tagged_sents) * 0.75)
    train_sents = tagged_sents[:split_idx]
    test_sents = tagged_sents[split_idx:]
    
    # 2. Build vocab and tags from training data only
    all_tags = sorted(list(set(str(t) for s in train_sents for w, t in s)))
    all_words = sorted(list(set(str(w).lower() for s in train_sents for w, t in s)))
    
    tag2idx = {tag: i for i, tag in enumerate(all_tags)}
    word2idx = {word: i for i, word in enumerate(all_words)}
    
    n_states, n_obs = len(all_tags), len(all_words)
    alpha = 0.001 # Smoothing
    
    # 3. Training: Calculate pi, P, and Q
    start_counts = np.zeros(n_states)
    trans_counts = np.zeros((n_states, n_states))
    emit_counts = np.zeros((n_states, n_obs))

    print(f"Training on {len(train_sents)} sentences...")
    for sent in train_sents:
        for i in range(len(sent)):
            w, t = str(sent[i][0]).lower(), str(sent[i][1])
            s_idx = tag2idx[t]
            o_idx = word2idx[w]
            if i == 0: start_counts[s_idx] += 1
            else:
                prev_t = str(sent[i-1][1])
                trans_counts[tag2idx[prev_t], s_idx] += 1
            emit_counts[s_idx, o_idx] += 1

    # Normalization
    start_p = (start_counts + alpha) / (np.sum(start_counts) + alpha * n_states)
    trans_p = (trans_counts + alpha) / (trans_counts.sum(axis=1, keepdims=True) + alpha * n_states)
    emit_p = (emit_counts + alpha) / (emit_counts.sum(axis=1, keepdims=True) + alpha * n_obs)

    # 4. Testing Accuracy on 25%
    print(f"Evaluating Viterbi on {len(test_sents)} sentences...")
    correct = 0
    total = 0
    
    # We test on a subset of the 25% if it's too slow, or use all for full accuracy
    for sent in test_sents[:500]: 
        noun_idx = tag2idx.get('NOUN', 0) 
        obs_indices = [word2idx.get(str(w).lower(), noun_idx) for w, t in sent]

        true_tags = [tag2idx.get(str(t), 0) for w, t in sent]
        
        # Run generic Viterbi (Using updated function name)
        predicted_path = sequence_algorithms.generic_viterbi_algorithm(obs_indices, start_p, trans_p, emit_p)
        
        for p, t in zip(predicted_path, true_tags):
            if p == t: correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"--- TRAINING COMPLETE ---")
    print(f"Viterbi Accuracy on Test Set: {accuracy:.2f}%")

    # 5. Save Model
    data = {
        'tags': all_tags, 'word2idx': word2idx,
        'start_p': start_p, 'trans_p': trans_p, 'emit_p': emit_p
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(data, f)

def load_pos_model():
    """Loads model from disk or trains if missing."""
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)