import os
import gc
import joblib
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from feature_engineering_v6 import (
    engineer_features_v6, build_trackman_meta_dict, build_pitcher_cluster_dict, 
    extract_zscore_stats
)

ID_COL = "row_id"
TARGET_COL = "control_success"

def clip_prob(pred):
    return np.clip(pred, 1e-5, 1 - 1e-5)

def calculate_bss(y_true, y_pred):
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0, 100000 * (1 - brier / baseline_brier))
    return score

class TabularTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_features, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(max(c+3, 4), d_model) for c in cat_cardinalities])
        self.num_weights = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.num_bias = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        
    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)
        cat_embs = [emb(torch.clamp(x_cat[:, i]+1, 0, emb.num_embeddings-1)).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        if len(cat_embs) > 0:
            cat_embs = torch.cat(cat_embs, dim=1)
        else:
            cat_embs = torch.empty(batch_size, 0, self.cls_token.size(-1), device=x_cat.device)
            
        num_embs = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_bias.unsqueeze(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        x = torch.cat([cls_tokens, cat_embs, num_embs], dim=1) 
        x = self.transformer(x)
        return self.head(x[:, 0, :])

class TabularWeightedDataset(Dataset):
    def __init__(self, X_cat, X_num, y, w):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.w = torch.tensor(w, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X_cat)
    def __getitem__(self, idx): return self.X_cat[idx], self.X_num[idx], self.y[idx], self.w[idx]

def train_v6():
    print("=== V6: GPU Full-Power Safe Pipeline (No Data Leakage) ===")
    
    BASE_DIR = os.environ.get("DATA_DIR", "./baseline/open")
    TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
    TRACKMAN_PATH = os.path.join(BASE_DIR, "data", "trackman_history.csv")
    
    meta_dict, _ = build_trackman_meta_dict(TRACKMAN_PATH)
    
    print("Loading train.csv...")
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    
    zscore_stats = extract_zscore_stats(df)
    pitcher_cluster_dict = build_pitcher_cluster_dict(df)
    
    print("Loading pitcher_trackman_stats...")
    TRACKMAN_STATS_PATH = os.path.join(BASE_DIR, "data", "pitcher_trackman_stats.pkl")
    if os.path.exists(TRACKMAN_STATS_PATH):
        trackman_stats_dict = joblib.load(TRACKMAN_STATS_PATH)
    else:
        trackman_stats_dict = None
    
    df = engineer_features_v6(df, meta_dict, pitcher_cluster_dict, zscore_stats, trackman_stats_dict)
    
    # 955점 돌파 1등 공신: 샘플 가중치
    weight_map = {2019: 0.2, 2020: 0.3, 2021: 0.4, 2022: 0.6, 2023: 0.8, 2024: 1.5}
    df["sample_weight"] = df["season"].map(weight_map).astype(np.float32)
    
    CAT_COLS = [
        "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
        "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
        "count_state", "pitcher_style_cluster", "stadium_owner_team"
    ]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [ID_COL, TARGET_COL, "season", "is_abs_era", "pitcher_id", "batter_id", "sample_weight", "H2H_matchup"] + CAT_COLS
    num_cols = [c for c in num_cols if c not in exclude]
    
    print(f"Total Categorical: {len(CAT_COLS)}, Numerical: {len(num_cols)}")
    
    cat_maps = {}
    cat_cardinalities = []
    for c in CAT_COLS:
        df[c] = df[c].fillna("Unknown").astype(str)
        unique_vals = df[c].unique()
        mapping = {v: i for i, v in enumerate(unique_vals)}
        cat_maps[c] = mapping
        df[c] = df[c].map(mapping)
        cat_cardinalities.append(len(unique_vals))
        
    num_medians = {}
    for c in num_cols:
        med = df[c].median()
        if pd.isna(med): med = 0.0
        num_medians[c] = med
        df[c] = df[c].fillna(med)
        
    X_cat = df[CAT_COLS].values.astype(np.int64)
    X_num = df[num_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    y = df[TARGET_COL].values.astype(np.float32)
    w = df["sample_weight"].values.astype(np.float32)
    
    # Global Target Encoding setup
    TE_COLS = ["pitcher_id", "batter_id"]
    global_te_maps = {}
    global_te_mean = float(df[TARGET_COL].mean())
    for c in TE_COLS:
        global_te_maps[c] = df.groupby(c)[TARGET_COL].mean().to_dict()
    
    # K-Fold OOF Predictions (shuffle=False to prevent Data Leakage)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    oof_xgb = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    oof_cat = np.zeros(len(df))
    oof_ftt = np.zeros(len(df))
    
    models_xgb = []
    models_lgb = []
    models_cat = []
    models_ftt = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FT-Transformer Device: {device}")
    
    start_fold = 0
    ckpt_path = os.path.join(BASE_DIR, "baseline_submit", "model", "v6_checkpoint_fold3.pkl")
    if os.path.exists(ckpt_path):
        print("Resuming from Fold 4...")
        ckpt = joblib.load(ckpt_path)
        models_xgb = ckpt["models_xgb"]
        models_lgb = ckpt["models_lgb"]
        models_cat = ckpt["models_cat"]
        models_ftt = ckpt["models_ftt"]
        start_fold = 3
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_tr_cat, X_va_cat = X_cat[train_idx], X_cat[val_idx]
        X_tr_num, X_va_num = X_num[train_idx], X_num[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        w_tr = w[train_idx]
        
        # OOF Target Encoding
        X_te_tr = np.zeros((len(train_idx), len(TE_COLS)), dtype=np.float32)
        X_te_va = np.zeros((len(val_idx), len(TE_COLS)), dtype=np.float32)
        
        for idx_te, c in enumerate(TE_COLS):
            tr_series = df.iloc[train_idx][c]
            va_series = df.iloc[val_idx][c]
            
            tr_map = pd.Series(y_tr).groupby(tr_series.values).mean().to_dict()
            tr_mean = float(y_tr.mean())
            
            X_te_tr[:, idx_te] = tr_series.map(tr_map).fillna(tr_mean).values
            X_te_va[:, idx_te] = va_series.map(tr_map).fillna(tr_mean).values
            
        te_scaler = StandardScaler()
        X_te_tr = te_scaler.fit_transform(X_te_tr)
        X_te_va = te_scaler.transform(X_te_va)
        
        X_tr_num_fold = np.hstack((X_tr_num, X_te_tr))
        X_va_num_fold = np.hstack((X_va_num, X_te_va))
        
        if fold < start_fold:
            print(f"Re-predicting OOF for Fold {fold+1} using loaded models...")
            dva = xgb.DMatrix(np.hstack((X_va_cat, X_va_num_fold)), label=y_va)
            oof_xgb[val_idx] = models_xgb[fold].predict(dva)
            oof_lgb[val_idx] = models_lgb[fold].predict(np.hstack((X_va_cat, X_va_num_fold)))
            oof_cat[val_idx] = models_cat[fold].predict_proba(np.hstack((X_va_cat, X_va_num_fold)))[:, 1]
            
            va_ds = TabularWeightedDataset(X_va_cat, X_va_num_fold, y_va, np.ones_like(y_va))
            va_loader = DataLoader(va_ds, batch_size=32768, shuffle=False)
            
            pt_model = TabularTransformer(cat_cardinalities, len(num_cols) + len(TE_COLS)).to(device)
            pt_model.load_state_dict(models_ftt[fold])
            pt_model.eval()
            
            va_preds = []
            with torch.no_grad():
                for xc, xn, _, _ in va_loader:
                    xc, xn = xc.to(device), xn.to(device)
                    preds = torch.sigmoid(pt_model(xc, xn)).cpu().numpy().flatten()
                    va_preds.extend(preds)
            oof_ftt[val_idx] = np.array(va_preds)
            print(f"Loaded OOF BSS - XGB: {calculate_bss(y_va, oof_xgb[val_idx]):.2f} | LGB: {calculate_bss(y_va, oof_lgb[val_idx]):.2f} | CAT: {calculate_bss(y_va, oof_cat[val_idx]):.2f} | FTT: {calculate_bss(y_va, oof_ftt[val_idx]):.2f}")
            continue
        
        # XGBoost
        dtr = xgb.DMatrix(np.hstack((X_tr_cat, X_tr_num_fold)), label=y_tr, weight=w_tr)
        dva = xgb.DMatrix(np.hstack((X_va_cat, X_va_num_fold)), label=y_va)
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'learning_rate': 0.05, 'max_depth': 6, 'tree_method': 'hist', 'device': 'cuda', 'n_jobs': 8}
        mxgb = xgb.train(xgb_params, dtr, num_boost_round=400, evals=[(dva, 'val')], early_stopping_rounds=20, verbose_eval=False)
        oof_xgb[val_idx] = mxgb.predict(dva)
        models_xgb.append(mxgb)
        
        # LightGBM (Disabled due to CPU deadlock)
        # ltr = lgb.Dataset(np.hstack((X_tr_cat, X_tr_num_fold)), label=y_tr, weight=w_tr)
        # lva = lgb.Dataset(np.hstack((X_va_cat, X_va_num_fold)), label=y_va, reference=ltr)
        # lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': 63, 'n_jobs': 8}
        # mlgb = lgb.train(lgb_params, ltr, num_boost_round=400, valid_sets=[lva], callbacks=[lgb.early_stopping(20, verbose=False)])
        oof_lgb[val_idx] = 0.5
        # models_lgb.append(mlgb)
        
        # CatBoost
        mcat = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=0, task_type='GPU', thread_count=8, loss_function='Logloss')
        mcat.fit(np.hstack((X_tr_cat, X_tr_num_fold)), y_tr, sample_weight=w_tr, eval_set=(np.hstack((X_va_cat, X_va_num_fold)), y_va), early_stopping_rounds=20)
        oof_cat[val_idx] = mcat.predict_proba(np.hstack((X_va_cat, X_va_num_fold)))[:, 1]
        models_cat.append(mcat)
        
        # FT-Transformer
        tr_ds = TabularWeightedDataset(X_tr_cat, X_tr_num_fold, y_tr, w_tr)
        va_ds = TabularWeightedDataset(X_va_cat, X_va_num_fold, y_va, np.ones_like(y_va))
        tr_loader = DataLoader(tr_ds, batch_size=4096, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=8192, shuffle=False)
        
        mftt = TabularTransformer(cat_cardinalities, len(num_cols) + len(TE_COLS)).to(device)
        optimizer = optim.AdamW(mftt.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        best_val_loss = float('inf')
        best_state = None
        for ep in range(60):
            mftt.train()
            for xc, xn, yb, wb in tr_loader:
                xc, xn, yb, wb = xc.to(device), xn.to(device), yb.to(device), wb.to(device)
                optimizer.zero_grad()
                preds = mftt(xc, xn)
                loss = (criterion(preds, yb) * wb).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            mftt.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for xc, xn, yb, wb in va_loader:
                    xc, xn, yb, wb = xc.to(device), xn.to(device), yb.to(device), wb.to(device)
                    preds = mftt(xc, xn)
                    loss = (criterion(preds, yb) * wb).mean()
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu() for k, v in mftt.state_dict().items()}
                
        mftt.load_state_dict(best_state)
                
        mftt.eval()
        va_preds = []
        with torch.no_grad():
            for xc, xn, _, _ in va_loader:
                xc, xn = xc.to(device), xn.to(device)
                preds = torch.sigmoid(mftt(xc, xn)).cpu().numpy().flatten()
                va_preds.extend(preds)
        oof_ftt[val_idx] = np.array(va_preds)
        
        # To avoid saving huge models in memory for 5 folds if not needed, we save state_dict
        mftt = mftt.cpu()
        models_ftt.append(mftt.state_dict())
        
        print(f"XGB BSS: {calculate_bss(y_va, oof_xgb[val_idx]):.2f} | LGB: {calculate_bss(y_va, oof_lgb[val_idx]):.2f} | CAT: {calculate_bss(y_va, oof_cat[val_idx]):.2f} | FTT: {calculate_bss(y_va, oof_ftt[val_idx]):.2f}")
        
        # Cleanup to prevent GPU OOM
        del dtr, dva, mftt, optimizer, criterion, tr_loader, va_loader, tr_ds, va_ds
        gc.collect()
        torch.cuda.empty_cache()
        
        # Checkpoint per fold
        fold_pipeline = {
            "models_xgb": models_xgb,
            "models_lgb": models_lgb,
            "models_cat": models_cat,
            "models_ftt": models_ftt
        }
        os.makedirs(os.path.join(BASE_DIR, "baseline_submit", "model"), exist_ok=True)
        joblib.dump(fold_pipeline, os.path.join(BASE_DIR, "baseline_submit", "model", f"v6_checkpoint_fold{fold+1}.pkl"), compress=3)

    # Isotonic Regression Calibration
    print("\n=== V6 Isotonic Regression Calibration ===")
    calibrators = {}
    
    cal_oof_xgb = np.zeros(len(y))
    iso_xgb = IsotonicRegression(out_of_bounds='clip')
    iso_xgb.fit(oof_xgb, y)
    cal_oof_xgb = iso_xgb.predict(oof_xgb)
    calibrators['xgb'] = iso_xgb
    
    cal_oof_lgb = np.zeros(len(y))
    # iso_lgb = IsotonicRegression(out_of_bounds='clip')
    # iso_lgb.fit(oof_lgb, y)
    # cal_oof_lgb = iso_lgb.predict(oof_lgb)
    # calibrators['lgb'] = iso_lgb
    
    cal_oof_cat = np.zeros(len(y))
    iso_cat = IsotonicRegression(out_of_bounds='clip')
    iso_cat.fit(oof_cat, y)
    cal_oof_cat = iso_cat.predict(oof_cat)
    calibrators['cat'] = iso_cat
    
    cal_oof_ftt = np.zeros(len(y))
    iso_ftt = IsotonicRegression(out_of_bounds='clip')
    iso_ftt.fit(oof_ftt, y)
    cal_oof_ftt = iso_ftt.predict(oof_ftt)
    calibrators['ftt'] = iso_ftt

    # Optimize weights
    print("\n=== V6 Ensemble Weight Optimization ===")
    preds_matrix = np.column_stack([cal_oof_xgb, cal_oof_cat, cal_oof_ftt])
    
    def loss_func(weights):
        weights = weights / np.sum(weights)
        ens = clip_prob(np.dot(preds_matrix, weights))
        brier = ((ens - y) ** 2).mean()
        return brier
        
    init_w = np.ones(3) / 3.0
    bounds = [(0, 1)] * 3
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)
    
    final_oof = clip_prob(np.dot(preds_matrix, best_weights))
    final_bss = calculate_bss(y, final_oof)
    
    print(f"Best Weights -> XGB: {best_weights[0]:.3f}, CAT: {best_weights[1]:.3f}, FTT: {best_weights[2]:.3f}")
    print(f"Final V6 OOF BSS: {final_bss:.2f}")

    MODEL_DIR = os.path.join(BASE_DIR, "baseline_submit", "model")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    pipeline = {
        "cat_cols": CAT_COLS,
        "num_cols": num_cols,
        "te_cols": TE_COLS,
        "global_te_maps": global_te_maps,
        "global_te_mean": global_te_mean,
        "cat_maps": cat_maps,
        "num_medians": num_medians,
        "scaler": scaler,
        "cat_cardinalities": cat_cardinalities,
        "trackman_meta_dict": meta_dict,
        "pitcher_cluster_dict": pitcher_cluster_dict,
        "zscore_stats": zscore_stats,
        "trackman_stats_dict": trackman_stats_dict,
        "models_xgb": models_xgb,
        "models_lgb": models_lgb,
        "models_cat": models_cat,
        "models_ftt": models_ftt,
        "calibrators": calibrators,
        "weights": best_weights.tolist()
    }
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "v6_pipeline.pkl"), compress=3)
    print("V6 Pipeline saved successfully.")

if __name__ == "__main__":
    train_v6()
