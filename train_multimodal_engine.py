import os
import time
import re
import numpy as np
import pandas as pd
import textstat
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def extract_linguistic_features_single(text, duration):
    text = str(text) if pd.notna(text) else ""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    n_words = len(words)
    dur_min = duration / 60.0 if duration > 0 else 1.0
    wpm = n_words / dur_min
    
    if n_words == 0:
        return {
            'ling_word_count': 0, 'ling_char_count': 0, 'ling_avg_word_length': 0.0,
            'ling_ttr': 0.0, 'ling_guiraud': 0.0, 'ling_hapax_ratio': 0.0, 'ling_long_word_ratio': 0.0,
            'ling_wpm': 0.0, 'ling_sent_count': 0, 'ling_avg_sent_len': 0.0,
            'ling_std_sent_len': 0.0, 'ling_max_sent_len': 0.0,
            'ling_modal_ratio': 0.0, 'ling_sub_conj': 0.0, 'ling_coord_conj': 0.0, 'ling_repetition_count': 0,
            'ling_fk_grade': 0.0, 'ling_fog': 0.0, 'ling_dale_chall': 0.0,
            'ling_reading_ease': 0.0, 'ling_coleman': 0.0, 'ling_ari': 0.0
        }
        
    unique_words = set(words)
    n_unique = len(unique_words)
    ttr = n_unique / n_words
    guiraud = n_unique / np.sqrt(n_words)
    
    word_freq = {}
    for w in words: word_freq[w] = word_freq.get(w, 0) + 1
    hapax_count = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax_count / n_words
    
    word_lens = [len(w) for w in words]
    avg_wlen = float(np.mean(word_lens))
    long_words = sum(1 for l in word_lens if l >= 6)
    long_word_ratio = long_words / n_words
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_count = max(len(sentences), 1)
    sent_lens = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences]
    avg_sent_len = float(np.mean(sent_lens)) if len(sent_lens) > 0 else float(n_words)
    std_sent_len = float(np.std(sent_lens)) if len(sent_lens) > 0 else 0.0
    max_sent_len = float(np.max(sent_lens)) if len(sent_lens) > 0 else float(n_words)
    
    modals = {'can', 'could', 'would', 'should', 'might', 'must', 'may', 'shall', 'ought'}
    sub_conjs = {'because', 'although', 'since', 'while', 'whereas', 'unless', 'though', 'if', 'even', 'whether', 'as'}
    coord_conjs = {'and', 'but', 'so', 'or', 'yet', 'for', 'nor'}
    
    modal_count = sum(1 for w in words if w in modals) / n_words
    sub_count = sum(1 for w in words if w in sub_conjs) / n_words
    coord_count = sum(1 for w in words if w in coord_conjs) / n_words
    
    reps = 0
    for i in range(len(words) - 1):
        if words[i] == words[i+1]:
            reps += 1
            
    try: fk_grade = float(textstat.flesch_kincaid_grade(text))
    except: fk_grade = 0.0
    try: fog = float(textstat.gunning_fog(text))
    except: fog = 0.0
    try: dc = float(textstat.dale_chall_readability_score(text))
    except: dc = 0.0
    try: ease = float(textstat.flesch_reading_ease(text))
    except: ease = 0.0
    try: coleman = float(textstat.coleman_liau_index(text))
    except: coleman = 0.0
    try: ari = float(textstat.automated_readability_index(text))
    except: ari = 0.0

    return {
        'ling_word_count': n_words,
        'ling_char_count': len(text),
        'ling_avg_word_length': avg_wlen,
        'ling_ttr': ttr,
        'ling_guiraud': guiraud,
        'ling_hapax_ratio': hapax_ratio,
        'ling_long_word_ratio': long_word_ratio,
        'ling_wpm': wpm,
        'ling_sent_count': sent_count,
        'ling_avg_sent_len': avg_sent_len,
        'ling_std_sent_len': std_sent_len,
        'ling_max_sent_len': max_sent_len,
        'ling_modal_ratio': modal_count,
        'ling_sub_conj': sub_count,
        'ling_coord_conj': coord_count,
        'ling_repetition_count': reps,
        'ling_fk_grade': fk_grade,
        'ling_fog': fog,
        'ling_dale_chall': dc,
        'ling_reading_ease': ease,
        'ling_coleman': coleman,
        'ling_ari': ari
    }

def build_multimodal_dataset():
    print("Loading acoustic features and transcripts...")
    train_ac = pd.read_csv('features_train.csv')
    test_ac = pd.read_csv('features_test.csv')
    
    train_tr = pd.read_csv('transcripts_train.csv')
    test_tr = pd.read_csv('transcripts_test.csv')
    
    train_merged = train_ac.merge(train_tr[['filename', 'transcript']], on='filename', how='left')
    test_merged = test_ac.merge(test_tr[['filename', 'transcript']], on='filename', how='left')
    
    print("Extracting linguistic & syntactic features...")
    train_ling = [extract_linguistic_features_single(row['transcript'], row['duration']) for _, row in train_merged.iterrows()]
    test_ling = [extract_linguistic_features_single(row['transcript'], row['duration']) for _, row in test_merged.iterrows()]
    
    train_ling_df = pd.DataFrame(train_ling)
    test_ling_df = pd.DataFrame(test_ling)
    
    # TF-IDF + TruncatedSVD for vocabulary semantics
    all_texts = train_merged['transcript'].fillna('').tolist() + test_merged['transcript'].fillna('').tolist()
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    svd = TruncatedSVD(n_components=16, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    
    n_tr = len(train_merged)
    train_svd = pd.DataFrame(svd_matrix[:n_tr], columns=[f'tfidf_svd_{i}' for i in range(16)])
    test_svd = pd.DataFrame(svd_matrix[n_tr:], columns=[f'tfidf_svd_{i}' for i in range(16)])
    
    drop_cols = ['filename', 'label', 'transcript']
    ac_cols = [c for c in train_ac.columns if c not in drop_cols]
    
    X_train = pd.concat([train_ac[ac_cols], train_ling_df, train_svd], axis=1)
    X_test = pd.concat([test_ac[ac_cols], test_ling_df, test_svd], axis=1)
    y_train = train_ac['label'].values
    
    print(f"Fused Multimodal Matrix: {X_train.shape[1]} features!")
    return X_train, y_train, X_test, train_ac, test_ac

def train_and_evaluate(X, y, X_test, train_df, test_df):
    medians = X.median()
    X = X.fillna(medians).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0)
    
    strat_labels = y.copy()
    strat_labels[strat_labels == 1.0] = 2.0
    strat_labels[strat_labels == 1.5] = 2.0
    strat_labels_cat = (strat_labels * 2).astype(int)
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {
        'CatBoost': {'oof': np.zeros(len(y)), 'test': np.zeros(len(X_test))},
        'XGB':      {'oof': np.zeros(len(y)), 'test': np.zeros(len(X_test))},
        'LGBM':     {'oof': np.zeros(len(y)), 'test': np.zeros(len(X_test))},
        'ET':       {'oof': np.zeros(len(y)), 'test': np.zeros(len(X_test))},
        'Ridge':    {'oof': np.zeros(len(y)), 'test': np.zeros(len(X_test))}
    }
    
    print("\n--- Training Multimodal 5-Fold Stratified Cross-Validation with CatBoost Super-Ensemble ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_labels_cat)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]
        
        # 1. CatBoost
        mc = CatBoostRegressor(iterations=750, learning_rate=0.03, depth=5, random_seed=42 + fold, verbose=0, thread_count=-1).fit(X_tr, y_tr)
        models['CatBoost']['oof'][val_idx] = mc.predict(X_va)
        models['CatBoost']['test'] += mc.predict(X_test) / n_splits
        
        # 2. XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=700, learning_rate=0.025, max_depth=5, subsample=0.8,
            colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.0, random_state=42 + fold,
            verbosity=0, n_jobs=-1
        ).fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
        models['XGB']['oof'][val_idx] = xgb_m.predict(X_va)
        models['XGB']['test'] += xgb_m.predict(X_test) / n_splits
        
        # 3. LightGBM
        lgb_m = lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.025, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42 + fold, verbose=-1, n_jobs=-1
        ).fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        models['LGBM']['oof'][val_idx] = lgb_m.predict(X_va)
        models['LGBM']['test'] += lgb_m.predict(X_test) / n_splits
        
        # 4. ExtraTrees
        et_m = ExtraTreesRegressor(
            n_estimators=350, max_depth=14, min_samples_split=4, max_features=0.55,
            random_state=42 + fold, n_jobs=-1
        ).fit(X_tr, y_tr)
        models['ET']['oof'][val_idx] = et_m.predict(X_va)
        models['ET']['test'] += et_m.predict(X_test) / n_splits
        
        # 5. Ridge
        ridge_pipe = Pipeline([('scaler', RobustScaler()), ('ridge', Ridge(alpha=15.0, random_state=42 + fold))]).fit(X_tr, y_tr)
        models['Ridge']['oof'][val_idx] = ridge_pipe.predict(X_va)
        models['Ridge']['test'] += ridge_pipe.predict(X_test) / n_splits

    # Optimal Ensembling
    m_keys = ['CatBoost', 'XGB', 'LGBM', 'ET', 'Ridge']
    oof_matrix = np.column_stack([np.clip(models[k]['oof'], 0.0, 5.0) for k in m_keys])
    test_matrix = np.column_stack([np.clip(models[k]['test'], 0.0, 5.0) for k in m_keys])
    
    def loss_fn(weights):
        w = np.array(weights) / np.sum(weights)
        pred = np.clip(oof_matrix @ w, 0.0, 5.0)
        return np.sqrt(mean_squared_error(y, pred))
        
    res = minimize(loss_fn, [0.50, 0.20, 0.10, 0.10, 0.10], method='SLSQP', bounds=[(0, 1)]*5, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    weights = res.x
    print(f"Optimal Blending Weights: {dict(zip(m_keys, np.round(weights, 3)))}")
    
    oof_ensemble = np.clip(oof_matrix @ weights, 0.0, 5.0)
    is_noise_train = (train_df['flat_mean'] > 0.40) & (train_df['cent_mean'] > 3500)
    oof_ensemble[is_noise_train] = 0.0
    
    val_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
    val_mae = mean_absolute_error(y, oof_ensemble)
    val_pearson, _ = pearsonr(y, oof_ensemble)
    val_spearman, _ = spearmanr(y, oof_ensemble)
    
    print("\n" + "="*55)
    print("=== MULTIMODAL OUT-OF-FOLD (VALIDATION) METRICS ===")
    print("="*55)
    print(f"Validation RMSE:               {val_rmse:.4f}")
    print(f"Validation Pearson Corr (r):   {val_pearson:.4f}  <-- Leaderboard Metric")
    print(f"Validation Leaderboard Loss:   {1 - val_pearson:.4f}  <-- Top rank is 0.3064")
    print(f"Validation MAE:                {val_mae:.4f}")
    print(f"Validation Spearman Corr (rho):{val_spearman:.4f}")
    print("="*55)
    
    # Train full model for mandatory Train RMSE
    final_cb = CatBoostRegressor(iterations=750, learning_rate=0.03, depth=5, random_seed=42, verbose=0, thread_count=-1).fit(X, y)
    final_xgb = xgb.XGBRegressor(n_estimators=400, learning_rate=0.025, max_depth=5, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=-1).fit(X, y)
    final_lgb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.025, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1).fit(X, y)
    final_et = ExtraTreesRegressor(n_estimators=350, max_depth=14, min_samples_split=4, max_features=0.55, random_state=42, n_jobs=-1).fit(X, y)
    final_ridge = Pipeline([('scaler', RobustScaler()), ('ridge', Ridge(alpha=15.0, random_state=42))]).fit(X, y)
    
    train_preds = (
        weights[0]*final_cb.predict(X) +
        weights[1]*final_xgb.predict(X) +
        weights[2]*final_lgb.predict(X) +
        weights[3]*final_et.predict(X) +
        weights[4]*final_ridge.predict(X)
    )
    train_preds[is_noise_train] = 0.0
    train_preds = np.clip(train_preds, 0.0, 5.0)
    
    train_rmse = np.sqrt(mean_squared_error(y, train_preds))
    print(f"\nTRAINING RMSE (COMPULSORY):    {train_rmse:.4f}")
    
    # Test Predictions
    test_preds = np.clip(test_matrix @ weights, 0.0, 5.0)
    is_noise_test = (test_df['flat_mean'] > 0.40) & (test_df['cent_mean'] > 3500)
    test_preds[is_noise_test] = 0.0
    
    sub = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.round(test_preds, 3)
    })
    sub.to_csv('submission.csv', index=False)
    sub.to_csv('Dataset_Final/test_predictions.csv', index=False)
    
    test_csv_updated = pd.read_csv('Dataset_Final/test.csv')
    test_map = dict(zip(sub['filename'], sub['label']))
    test_csv_updated['label'] = test_csv_updated['filename'].map(test_map)
    test_csv_updated.to_csv('Dataset_Final/test.csv', index=False)
    
    print("\nUpdated submission.csv and Dataset_Final/test.csv successfully!")
    print(sub.head(10))

if __name__ == '__main__':
    X_tr, y_tr, X_te, tr_df, te_df = build_multimodal_dataset()
    train_and_evaluate(X_tr, y_tr, X_te, tr_df, te_df)
