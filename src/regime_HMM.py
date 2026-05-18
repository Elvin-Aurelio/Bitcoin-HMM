import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

class RegimeHMM:
    """
    Pembungkus Gaussian HMM dengan fitur:
    1. Warm Start untuk kontinuitas konvergensi.
    2. Label Tracking menggunakan Euclidean Distance & Hungarian Algorithm.
    """
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.state_mapping = {}
        
        # Variabel memori
        self.is_fitted = False
        self.reference_means = None # Menyimpan vektor profil ideal tiap label
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None

    def fit(self, df_train, feature_cols, warm_start=True):
        X_train = df_train[feature_cols].values
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # --- 1. TRAINING (WARM START VS COLD START) ---
        if warm_start and self.is_fitted:
            self.model = GaussianHMM(n_components=self.n_components, 
                                     covariance_type="full", n_iter=100,
                                     init_params="", random_state=self.random_state)
            self.model.startprob_ = self.startprob_
            self.model.transmat_ = self.transmat_
            self.model.means_ = self.means_
            self.model.covars_ = self.covars_
        else:
            self.model = GaussianHMM(n_components=self.n_components, 
                                     covariance_type="full", n_iter=100, 
                                     init_params="stmc", random_state=self.random_state)
            
        self.model.fit(X_train_scaled)
        current_means = self.model.means_
        
        # --- 2. STATE MAPPING ---
        if not self.is_fitted:
            # COLD START: Definisikan identitas awal (hanya sekali)
            # Pakai logika Risk/Reward (Return / Volatility) untuk set pondasi awal
            ret_idx = next(i for i, col in enumerate(feature_cols) if 'return' in col.lower())
            vol_idx = next(i for i, col in enumerate(feature_cols) if 'vol' in col.lower() or 'range' in col.lower())
            
            state_stats = []
            for i in range(self.n_components):
                score = current_means[i][ret_idx] / (current_means[i][vol_idx] + 1e-8)
                state_stats.append({'state': i, 'score': score})
                
            sorted_by_score = sorted(state_stats, key=lambda x: x['score'])
            
            self.state_mapping = {
                sorted_by_score[0]['state']: 0,  # 0 = Crash (Terburuk)
                sorted_by_score[1]['state']: 1,  # 1 = Sideways
                sorted_by_score[2]['state']: 2   # 2 = Bull (Terbaik)
            }
            
            # Simpan vektor pusat (means) sebagai "Buku Referensi"
            self.reference_means = np.zeros_like(current_means)
            for raw_state, logical_label in self.state_mapping.items():
                self.reference_means[logical_label] = current_means[raw_state]
                
        else:
            # Hitung matriks jarak Euclidean antara State Baru vs State Referensi Lama
            # Ukuran matriks: (3, 3)
            dist_matrix = np.zeros((self.n_components, self.n_components))
            for i in range(self.n_components):
                for j in range(self.n_components):
                    dist_matrix[i, j] = euclidean(current_means[i], self.reference_means[j])
            
            # Gunakan Hungarian Algorithm untuk pemetaan 1-1 terbaik.
            # Algoritma mencari "Biaya Minimum", sehingga jarak yang lebih kecil diprioritaskan.
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            # row_ind adalah index state mentah yang baru.
            # col_ind adalah label logika referensi (0=Crash, 1=Sideways, 2=Bull).
            new_mapping = {}
            for raw_state, logical_label in zip(row_ind, col_ind):
                new_mapping[raw_state] = logical_label
                
                # Update "Buku Referensi" agar bisa melacak pergeseran pasar pelan-pelan (Drift Tracking)
                # 80% Buku Referensi Lama, 20% Profil Baru (dengan bobot lebih kecil agar tidak mudah terombang-ambing oleh noise)
                alpha = 0.2
                self.reference_means[logical_label] = (alpha * current_means[raw_state]) + ((1 - alpha) * self.reference_means[logical_label])
                
            self.state_mapping = new_mapping

        # Simpan memori untuk iterasi berikutnya
        self.is_fitted = True
        self.startprob_ = self.model.startprob_
        self.transmat_ = self.model.transmat_
        self.means_ = self.model.means_
        self.covars_ = self.model.covars_
        
        return self

    def predict(self, df_test, feature_cols):
        X_test = df_test[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        raw_states = self.model.predict(X_test_scaled)
        mapped_states = [self.state_mapping[s] for s in raw_states]
        return mapped_states
        
    def predict_proba(self, df_test, feature_cols):
        X_test = df_test[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        raw_proba = self.model.predict_proba(X_test_scaled)
        
        # Reorder columns to match logical state labels
        remapped_proba = np.zeros_like(raw_proba)
        for raw_state, logical_label in self.state_mapping.items():
            remapped_proba[:, logical_label] = raw_proba[:, raw_state]
        
        return remapped_proba