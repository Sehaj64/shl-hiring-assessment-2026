import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def compute_qwk(y_true, y_pred):
    # Quantize to 0.5 increments for QWK
    classes = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    y_true_q = np.array([classes[np.abs(classes - v).argmin()] for v in y_true])
    y_pred_q = np.array([classes[np.abs(classes - v).argmin()] for v in y_pred])
    
    # Map classes to integers 0..9
    mapping = {c: i for i, c in enumerate(classes)}
    y_t_int = np.array([mapping[v] for v in y_true_q])
    y_p_int = np.array([mapping[v] for v in y_pred_q])
    return cohen_kappa_score(y_t_int, y_p_int, weights='quadratic')

def evaluate_predictions(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    qwk = compute_qwk(y_true, y_pred)
    print(f"[{name}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | Pearson: {pearson_corr:.4f} | Spearman: {spearman_corr:.4f} | QWK: {qwk:.4f}")
    return rmse, mae, pearson_corr, spearman_corr, qwk

def main():
    print("Loading extracted feature datasets...")
    train_df = pd.read_csv('features_train.csv')
    test_df = pd.read_csv('features_test.csv')
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Feature columns
    drop_cols = ['filename', 'label']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df['label'].values
    X_test = test_df[feature_cols].copy()
    
    # Handle any potential NaNs or infs by median imputation
    medians = X.median()
    X = X.fillna(medians).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0)
    
    # Stratified K-Fold using discrete labels
    # Group rare classes (1.0, 1.5) with 2.0 for stratification fold split
    strat_labels = y.copy()
    strat_labels[strat_labels == 1.0] = 2.0
    strat_labels[strat_labels == 1.5] = 2.0
    strat_labels_cat = (strat_labels * 2).astype(int)
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store out-of-fold and test predictions for each model family
    models = {
        'LGBM': {
            'oof': np.zeros(len(y)),
            'test': np.zeros(len(X_test)),
        },
        'XGB': {
            'oof': np.zeros(len(y)),
            'test': np.zeros(len(X_test)),
        },
        'ET': {
            'oof': np.zeros(len(y)),
            'test': np.zeros(len(X_test)),
        },
        'Ridge': {
            'oof': np.zeros(len(y)),
            'test': np.zeros(len(X_test)),
        }
    }
    
    print(f"\n--- Starting 5-Fold Stratified Cross-Validation on {len(feature_cols)} features ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_labels_cat)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]
        
        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        val_pred_lgb = lgb_model.predict(X_va)
        models['LGBM']['oof'][val_idx] = val_pred_lgb
        models['LGBM']['test'] += lgb_model.predict(X_test) / n_splits
        
        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            verbose=False
        )
        val_pred_xgb = xgb_model.predict(X_va)
        models['XGB']['oof'][val_idx] = val_pred_xgb
        models['XGB']['test'] += xgb_model.predict(X_test) / n_splits
        
        # 3. ExtraTrees
        et_model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            max_features=0.6,
            random_state=42 + fold,
            n_jobs=-1
        )
        et_model.fit(X_tr, y_tr)
        val_pred_et = et_model.predict(X_va)
        models['ET']['oof'][val_idx] = val_pred_et
        models['ET']['test'] += et_model.predict(X_test) / n_splits
        
        # 4. Ridge Regression with RobustScaler
        ridge_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('ridge', Ridge(alpha=10.0, random_state=42 + fold))
        ])
        ridge_pipe.fit(X_tr, y_tr)
        val_pred_ridge = ridge_pipe.predict(X_va)
        models['Ridge']['oof'][val_idx] = val_pred_ridge
        models['Ridge']['test'] += ridge_pipe.predict(X_test) / n_splits
        
    print("\n=== Out-Of-Fold Single Model Results ===")
    for mname in models:
        models[mname]['oof'] = np.clip(models[mname]['oof'], 0.0, 5.0)
        evaluate_predictions(y, models[mname]['oof'], name=mname)
        
    # Find Optimal Blending Weights
    print("\n=== Optimizing Ensemble Weights via Nelder-Mead ===")
    m_keys = ['LGBM', 'XGB', 'ET', 'Ridge']
    oof_matrix = np.column_stack([models[k]['oof'] for k in m_keys])
    test_matrix = np.column_stack([models[k]['test'] for k in m_keys])
    
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.clip(oof_matrix @ w, 0.0, 5.0)
        return np.sqrt(mean_squared_error(y, pred))
        
    init_weights = [0.4, 0.3, 0.2, 0.1]
    bounds = [(0, 1)] * len(m_keys)
    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    best_weights = res.x
    print(f"Optimal Ensemble Weights: {dict(zip(m_keys, np.round(best_weights, 3)))}")
    
    oof_ensemble = np.clip(oof_matrix @ best_weights, 0.0, 5.0)
    print("\n=== Final Blended Ensemble Out-Of-Fold Evaluation ===")
    evaluate_predictions(y, oof_ensemble, name="Ensemble Blend")
    
    # Static / Silent Audio Detector Override
    # Files with extreme spectral flatness > 0.40 and centroid > 3500 are artificial static (score 0.0)
    is_noise_train = (train_df['flat_mean'] > 0.40) & (train_df['cent_mean'] > 3500)
    oof_ensemble_post = oof_ensemble.copy()
    oof_ensemble_post[is_noise_train] = 0.0
    print("\n=== Ensemble with Static/Noise Detection Rule ===")
    evaluate_predictions(y, oof_ensemble_post, name="Ensemble + Noise Rule")
    
    # Generate Predictions for Test Set
    test_preds = np.clip(test_matrix @ best_weights, 0.0, 5.0)
    is_noise_test = (test_df['flat_mean'] > 0.40) & (test_df['cent_mean'] > 3500)
    test_preds[is_noise_test] = 0.0
    
    # Discrete round grid mapping for assessment format
    classes = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    test_preds_rounded = np.array([classes[np.abs(classes - v).argmin()] for v in test_preds])
    
    # Save predictions
    sub_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.round(test_preds, 3),
        'label_rounded': test_preds_rounded
    })
    sub_df.to_csv('test_predictions_detailed.csv', index=False)
    
    # Save standard test.csv submission
    # Many competitions accept continuous regression predictions or rounded
    # Let's write the exact continuous high-precision prediction into test_predictions.csv
    # and update Dataset_Final/test.csv
    submission_continuous = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.round(test_preds, 2)
    })
    submission_continuous.to_csv('Dataset_Final/test_predictions.csv', index=False)
    submission_continuous.to_csv('test_predictions.csv', index=False)
    
    submission_rounded = pd.DataFrame({
        'filename': test_df['filename'],
        'label': test_preds_rounded
    })
    submission_rounded.to_csv('Dataset_Final/test_predictions_rounded.csv', index=False)
    
    # Also update Dataset_Final/test.csv with the predicted labels
    test_csv_updated = pd.read_csv('Dataset_Final/test.csv')
    test_pred_map = dict(zip(sub_df['filename'], sub_df['label']))
    test_csv_updated['label'] = test_csv_updated['filename'].map(test_pred_map)
    test_csv_updated.to_csv('Dataset_Final/test.csv', index=False)
    
    # Also check sample_submission.csv format
    # In sample_submission.csv, update any files that appear in test
    sample_sub = pd.read_csv('Dataset_Final/sample_submission.csv')
    sample_sub['label'] = sample_sub['filename'].map(test_pred_map).fillna(sample_sub['label'])
    # If any file is in train, fill with its train label
    train_label_map = dict(zip(train_df['filename'], train_df['label']))
    sample_sub['label'] = sample_sub['filename'].map(train_label_map).fillna(sample_sub['label'])
    sample_sub.to_csv('Dataset_Final/sample_submission_filled.csv', index=False)
    
    print("\n=== Test Predictions Summary ===")
    print(sub_df.head(15))
    print("\nTest predictions distribution:")
    print(pd.Series(test_preds_rounded).value_counts().sort_index())
    print("\nSaved predictions to:")
    print(" - Dataset_Final/test.csv (updated with predictions)")
    print(" - Dataset_Final/test_predictions.csv")
    print(" - Dataset_Final/test_predictions_rounded.csv")
    print(" - Dataset_Final/sample_submission_filled.csv")
    print(" - test_predictions_detailed.csv")
    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
