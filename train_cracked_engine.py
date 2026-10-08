import os
import time
import re
import numpy as np
import pandas as pd
import textstat
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
import warnings
warnings.filterwarnings('ignore')

print("="*65)
print("=== ADVANCED CRACKED MULTIMODAL GRAMMAR SCORING ENGINE ===")
print("="*65)

# 1. Load Data
train_ac = pd.read_csv('features_train.csv')
test_ac = pd.read_csv('features_test.csv')
train_tr = pd.read_csv('transcripts_train.csv')
test_tr = pd.read_csv('transcripts_test.csv')

train_merged = train_ac.merge(train_tr[['filename', 'transcript']], on='filename', how='left')
test_merged = test_ac.merge(test_tr[['filename', 'transcript']], on='filename', how='left')

# 2. Rich Linguistic, Syntactic & Error Features
def extract_linguistics(text, dur):
    text = str(text) if pd.notna(text) else ""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    n = len(words)
    dur_min = dur / 60.0 if dur > 0 else 1.0
    wpm = n / dur_min
    
    unique_words = set(words)
    u = len(unique_words)
    word_lens = [len(w) for w in words]
    long_words = sum(1 for l in word_lens if l >= 6)
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sent = max(len(sentences), 1)
    sent_lens = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences]
    
    # Fillers & repetitions
    fillers = {'um', 'uh', 'er', 'ah', 'like', 'yeah', 'basically', 'actually', 'mean'}
    n_fillers = sum(1 for w in words if w in fillers)
    reps = sum(1 for i in range(n - 1) if words[i] == words[i+1])
    
    # Syntactic & grammar error patterns
    sv_patterns = [
        r'\b(he|she|it)\s+(don\'t|are|were|have)\b',
        r'\b(i|we|they|you)\s+(is|was|has)\b',
        r'\b(there)\s+(is|was)\s+(\d+|many|several|a\s+lot\s+of|two|three)\b',
        r'\b(a)\s+[aeiou]\w+\b',
        r'\b(an)\s+[bcdfghjklmnpqrstvwxyz]\w+\b',
        r'\b(more)\s+(better|faster|harder|easier|bigger)\b'
    ]
    errs = sum(len(re.findall(p, text.lower())) for p in sv_patterns)
    
    res = {
        'ling_wpm': wpm,
        'ling_word_count': n,
        'ling_char_count': len(text),
        'ling_ttr': u / max(n, 1),
        'ling_guiraud': u / np.sqrt(max(n, 1)),
        'ling_avg_word_len': float(np.mean(word_lens)) if n > 0 else 0.0,
        'ling_long_word_r': long_words / max(n, 1),
        'ling_sent_count': n_sent,
        'ling_avg_sent_len': float(np.mean(sent_lens)) if sent_lens else float(n),
        'ling_filler_ratio': n_fillers / max(n, 1),
        'ling_rep_ratio': reps / max(n, 1),
        'ling_err_ratio': errs / max(n, 1),
    }
    
    try: res['ling_fk'] = float(textstat.flesch_kincaid_grade(text))
    except: res['ling_fk'] = 0.0
    try: res['ling_fog'] = float(textstat.gunning_fog(text))
    except: res['ling_fog'] = 0.0
    try: res['ling_read_ease'] = float(textstat.flesch_reading_ease(text))
    except: res['ling_read_ease'] = 0.0
    
    # POS tagging
    tokens = word_tokenize(text)
    tags = [t for w, t in pos_tag(tokens) if w.isalnum()]
    n_pos = len(tags)
    if n_pos > 0:
        nouns = sum(1 for t in tags if t.startswith('NN'))
        verbs = sum(1 for t in tags if t.startswith('VB'))
        adjs = sum(1 for t in tags if t.startswith('JJ'))
        advs = sum(1 for t in tags if t.startswith('RB'))
        prons = sum(1 for t in tags if t.startswith('PRP'))
        conjs = sum(1 for t in tags if t in ('CC', 'IN'))
        res['pos_noun_r'] = nouns / n_pos
        res['pos_verb_r'] = verbs / n_pos
        res['pos_adj_r'] = adjs / n_pos
        res['pos_adv_r'] = advs / n_pos
        res['pos_pron_r'] = prons / n_pos
        res['pos_conj_r'] = conjs / n_pos
        res['pos_pron_to_noun'] = prons / (nouns + 1e-4)
        res['pos_distinct'] = len(set(tags)) / 36.0
    else:
        for k in ['pos_noun_r', 'pos_verb_r', 'pos_adj_r', 'pos_adv_r', 'pos_pron_r', 'pos_conj_r', 'pos_pron_to_noun', 'pos_distinct']:
            res[k] = 0.0
    return res

print("Extracting linguistic & syntactic features...")
train_ling = pd.DataFrame([extract_linguistics(r['transcript'], r['duration']) for _, r in train_merged.iterrows()])
test_ling = pd.DataFrame([extract_linguistics(r['transcript'], r['duration']) for _, r in test_merged.iterrows()])

# 3. Dense Neural SBERT Embeddings (32 PCA)
n_tr = len(train_merged)
embs = np.load('scratch_embeddings.npy')
pca = PCA(n_components=32, random_state=42)
embs_pca = pca.fit_transform(embs)
train_emb = pd.DataFrame(embs_pca[:n_tr], columns=[f'emb_pca_{i}' for i in range(32)])
test_emb = pd.DataFrame(embs_pca[n_tr:], columns=[f'emb_pca_{i}' for i in range(32)])

# 4. Multimodal Acoustic Interactions
drop_cols = ['filename', 'label', 'transcript']
ac_cols = [c for c in train_ac.columns if c not in drop_cols]

def add_interactions(df_ac, df_ling):
    inter = pd.DataFrame()
    inter['fluency_lexical_product'] = df_ling['ling_wpm'] * df_ling['ling_guiraud']
    inter['speech_ratio_per_sent'] = df_ac['speech_ratio'] / (df_ling['ling_sent_count'] + 1)
    inter['mfcc2_guiraud_product'] = df_ac['mfcc_2_std'] * df_ling['ling_guiraud']
    inter['loudness_dynamic_range'] = df_ac['loudness_sma3_percentile80.0'] - df_ac['loudness_sma3_percentile20.0']
    inter['jitter_shimmer_ratio'] = df_ac['jitterLocal_sma3nz_amean'] / (df_ac['shimmerLocaldB_sma3nz_amean'] + 1e-5)
    return inter

train_inter = add_interactions(train_ac, train_ling)
test_inter = add_interactions(test_ac, test_ling)

X_train = pd.concat([train_ac[ac_cols], train_ling, train_inter, train_emb], axis=1)
X_test = pd.concat([test_ac[ac_cols], test_ling, test_inter, test_emb], axis=1)
y_train = train_ac['label'].values

medians = X_train.median()
X_train = X_train.fillna(medians).replace([np.inf, -np.inf], 0)
X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0)

print(f"Total Enhanced Features: {X_train.shape[1]}")

# 5. Model Suite & 10-Fold Stratified CV
strat = y_train.copy()
strat[strat <= 1.5] = 2.0
strat_cat = (strat * 2).astype(int)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

models = {
    'CatBoost': {'oof': np.zeros(len(y_train)), 'test': np.zeros(len(X_test))},
    'LGBM':     {'oof': np.zeros(len(y_train)), 'test': np.zeros(len(X_test))},
    'XGB':      {'oof': np.zeros(len(y_train)), 'test': np.zeros(len(X_test))},
    'Ridge':    {'oof': np.zeros(len(y_train)), 'test': np.zeros(len(X_test))},
    'ET':       {'oof': np.zeros(len(y_train)), 'test': np.zeros(len(X_test))}
}

print(f"\n--- Training 10-Fold Stratified Super-Ensemble ---")
t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, strat_cat)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    # 1. CatBoost
    cb = CatBoostRegressor(iterations=750, learning_rate=0.03, depth=5, l2_leaf_reg=3, random_seed=42+fold, verbose=0, thread_count=-1).fit(X_tr, y_tr)
    models['CatBoost']['oof'][val_idx] = cb.predict(X_va)
    models['CatBoost']['test'] += cb.predict(X_test) / n_splits
    
    # 2. LightGBM
    lgb_m = lgb.LGBMRegressor(n_estimators=650, learning_rate=0.025, max_depth=5, num_leaves=24, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.5, random_state=42+fold, verbose=-1, n_jobs=-1).fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    models['LGBM']['oof'][val_idx] = lgb_m.predict(X_va)
    models['LGBM']['test'] += lgb_m.predict(X_test) / n_splits
    
    # 3. XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=650, learning_rate=0.025, max_depth=4, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.5, random_state=42+fold, verbosity=0, n_jobs=-1).fit(X_tr, y_tr, eval_set=[(X_va, y_val)], verbose=False)
    models['XGB']['oof'][val_idx] = xgb_m.predict(X_va)
    models['XGB']['test'] += xgb_m.predict(X_test) / n_splits
    
    # 4. Ridge Pipeline
    rp = Pipeline([('scaler', RobustScaler()), ('ridge', Ridge(alpha=18.0, random_state=42+fold))]).fit(X_tr, y_tr)
    models['Ridge']['oof'][val_idx] = rp.predict(X_va)
    models['Ridge']['test'] += rp.predict(X_test) / n_splits
    
    # 5. ExtraTrees
    et = ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=4, max_features=0.45, random_state=42+fold, n_jobs=-1).fit(X_tr, y_tr)
    models['ET']['oof'][val_idx] = et.predict(X_va)
    models['ET']['test'] += et.predict(X_test) / n_splits

print(f"10-Fold Training Completed in {time.time()-t0:.1f}s")

# 6. SLSQP Optimal Out-of-Fold Blending
m_keys = ['CatBoost', 'LGBM', 'XGB', 'Ridge', 'ET']
oof_mat = np.column_stack([np.clip(models[k]['oof'], 0.0, 5.0) for k in m_keys])
test_mat = np.column_stack([np.clip(models[k]['test'], 0.0, 5.0) for k in m_keys])

def blend_loss(weights):
    w = np.array(weights) / np.sum(weights)
    pred = np.clip(oof_mat @ w, 0.0, 5.0)
    return np.sqrt(mean_squared_error(y_train, pred))

res = minimize(blend_loss, [0.35, 0.25, 0.15, 0.20, 0.05], method='SLSQP', bounds=[(0, 1)]*5, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
weights = res.x
print("\nOptimal Model Blending Weights:", dict(zip(m_keys, np.round(weights, 3))))

oof_blended = np.clip(oof_mat @ weights, 0.0, 5.0)
test_blended = np.clip(test_mat @ weights, 0.0, 5.0)

# Static noise override
is_noise_train = (train_ac['flat_mean'] > 0.40) & (train_ac['cent_mean'] > 3500)
oof_blended[is_noise_train] = 0.0

raw_rmse = np.sqrt(mean_squared_error(y_train, oof_blended))
raw_mae = mean_absolute_error(y_train, oof_blended)
raw_r, _ = pearsonr(y_train, oof_blended)
raw_rho, _ = spearmanr(y_train, oof_blended)

print("\n" + "="*55)
print("=== OUT-OF-FOLD (VALIDATION) METRICS (RAW BLEND) ===")
print(f"Validation RMSE:               {raw_rmse:.4f}")
print(f"Validation Pearson Corr (r):   {raw_r:.4f}")
print(f"Validation Leaderboard Loss:   {1 - raw_r:.4f}")
print(f"Validation MAE:                {raw_mae:.4f}")
print(f"Validation Spearman Corr:      {raw_rho:.4f}")
print("="*55)

# 7. Post-Processing Calibration Optimization
# Optimize scale alpha and intercept beta on out-of-fold speech to minimize RMSE directly!
speech_mask_train = ~is_noise_train
speech_y = y_train[speech_mask_train]
speech_pred = oof_blended[speech_mask_train]

def calib_loss(params):
    alpha, beta = params
    p = np.clip(alpha * speech_pred + beta, 1.0, 5.0)
    return np.sqrt(mean_squared_error(speech_y, p))

calib_res = minimize(calib_loss, [1.0, 0.0], method='Nelder-Mead')
best_alpha, best_beta = calib_res.x
print(f"\nOptimized Calibration Parameters: alpha = {best_alpha:.4f}, beta = {best_beta:.4f}")

oof_calibrated = oof_blended.copy()
oof_calibrated[speech_mask_train] = np.clip(best_alpha * speech_pred + best_beta, 1.0, 5.0)
calib_rmse = np.sqrt(mean_squared_error(y_train, oof_calibrated))
print(f"Calibrated Validation RMSE:    {calib_rmse:.4f} (improvement: {raw_rmse - calib_rmse:.4f})")

# 8. Test Set Predictions & Distribution Alignment
test_preds = test_blended.copy()
is_noise_test = (test_ac['flat_mean'] > 0.40) & (test_ac['cent_mean'] > 3500)
test_preds[is_noise_test] = 0.0

# Apply calibration on valid speech test predictions
test_preds[~is_noise_test] = np.clip(best_alpha * test_preds[~is_noise_test] + best_beta, 1.0, 5.0)

# Quantile smoothing with training valid distribution
valid_train_y_sorted = np.sort(speech_y)
test_ranks = np.argsort(np.argsort(test_preds)) / len(test_preds)
test_quantiles = np.quantile(valid_train_y_sorted, test_ranks)

# Blended calibration: 70% optimized regression + 30% empirical quantile anchor
final_test_preds = np.clip(0.70 * test_preds + 0.30 * test_quantiles, 1.0, 5.0)
final_test_preds[is_noise_test] = 0.0

# Save final predictions
sub = pd.DataFrame({
    'filename': test_ac['filename'],
    'label': np.round(final_test_preds, 3)
})
sub.to_csv('submission.csv', index=False)
sub.to_csv('Dataset_Final/test_predictions.csv', index=False)

test_csv = pd.read_csv('Dataset_Final/test.csv')
test_map = dict(zip(sub['filename'], sub['label']))
test_csv['label'] = test_csv['filename'].map(test_map)
test_csv.to_csv('Dataset_Final/test.csv', index=False)

print("\nFinal Test Predictions Summary:")
print(sub.head(15))
print("\nFinal Test Predictions Distribution:")
print(sub['label'].describe())

# Mandatory Training Set Evaluation
final_cb = CatBoostRegressor(iterations=750, learning_rate=0.03, depth=5, l2_leaf_reg=3, random_seed=42, verbose=0, thread_count=-1).fit(X_train, y_train)
final_lgb = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.025, max_depth=5, num_leaves=24, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.5, random_state=42, verbose=-1, n_jobs=-1).fit(X_train, y_train)
final_xgb = xgb.XGBRegressor(n_estimators=450, learning_rate=0.025, max_depth=4, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.1, reg_lambda=1.5, random_state=42, verbosity=0, n_jobs=-1).fit(X_train, y_train)
final_ridge = Pipeline([('scaler', RobustScaler()), ('ridge', Ridge(alpha=18.0, random_state=42))]).fit(X_train, y_train)
final_et = ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=4, max_features=0.45, random_state=42, n_jobs=-1).fit(X_train, y_train)

train_preds = (
    weights[0] * final_cb.predict(X_train) +
    weights[1] * final_lgb.predict(X_train) +
    weights[2] * final_xgb.predict(X_train) +
    weights[3] * final_ridge.predict(X_train) +
    weights[4] * final_et.predict(X_train)
)
train_preds[is_noise_train] = 0.0
train_preds = np.clip(train_preds, 0.0, 5.0)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mae = mean_absolute_error(y_train, train_preds)
train_r, _ = pearsonr(y_train, train_preds)

print("\n" + "*"*60)
print(">>> MANDATORY ASSESSMENT EVALUATION REQUIREMENT <<<")
print(f"TRAINING RMSE (COMPULSORY):    {train_rmse:.4f}")
print(f"TRAINING MAE:                  {train_mae:.4f}")
print(f"TRAINING PEARSON (r):          {train_r:.4f}")
print("*"*60)
print("\nCracked engine execution finished successfully!")
