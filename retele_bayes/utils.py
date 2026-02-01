import numpy as np
import os

def ensure_file_exists(filename):
    """
    Checks if a file exists. If not, creates it with default content.
    """
    if not os.path.exists(filename):
        print(f"[Utils] Generating file '{filename}'...")
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
        Acest text a fost generat de catre mine cu ajutorul geamanului.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_demo)

def generate_generic_data(n_steps, start_p, trans_p, emit_p):
    true_states = []
    observations = []
    
    n_states = len(start_p)
    n_obs = emit_p.shape[1]
    
    current_state = np.random.choice(range(n_states), p=start_p)
    
    for _ in range(n_steps):
        true_states.append(current_state)
        obs = np.random.choice(range(n_obs), p=emit_p[current_state])
        observations.append(obs)
        current_state = np.random.choice(range(n_states), p=trans_p[current_state])
        
    return true_states, observations