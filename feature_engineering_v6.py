import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

DATA_DIR = "./baseline/open/data"
HAND_MAP = {"Right": 2, "Left": 1}

META_METRIC_COLS = [
    "tm_rel_speed_mean", "tm_rel_speed_std",
    "tm_spin_rate_mean", "tm_spin_rate_std",
    "tm_induced_vert_break_mean",
    "tm_horz_break_mean",
    "tm_extension_mean",
    "tm_rel_height_mean",
    "tm_rel_side_mean"
]

def build_trackman_meta_dict(trackman_path):
    print("Loading trackman_history.csv for meta aggregation...")
    usecols = [
        "season", "pitcher_hand", "batter_hand", "rel_speed", "spin_rate", 
        "induced_vert_break", "horz_break", "extension", 
        "rel_height", "rel_side"
    ]
    df = pd.read_csv(trackman_path, usecols=usecols)
    
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    df.dropna(subset=["pitcher_hand", "batter_hand"], inplace=True)
    df["pitcher_hand"] = df["pitcher_hand"].astype(int)
    df["batter_hand"] = df["batter_hand"].astype(int)
    
    agg_funcs = {
        "rel_speed": ["mean", "std"],
        "spin_rate": ["mean", "std"],
        "induced_vert_break": ["mean"],
        "horz_break": ["mean"],
        "extension": ["mean"],
        "rel_height": ["mean"],
        "rel_side": ["mean"]
    }
    
    meta_stats = df.groupby(["season", "pitcher_hand", "batter_hand"]).agg(agg_funcs)
    meta_stats.columns = META_METRIC_COLS
    meta_stats.reset_index(inplace=True)
    
    meta_2024 = meta_stats[meta_stats["season"] == 2024].copy()
    meta_2024["season"] = 2025
    meta_stats_full = pd.concat([meta_stats, meta_2024], ignore_index=True)
    
    meta_dict = {}
    for _, row in meta_stats_full.iterrows():
        key = (int(row["season"]), int(row["pitcher_hand"]), int(row["batter_hand"]))
        vals = [float(row[c]) if not np.isnan(row[c]) else 0.0 for c in META_METRIC_COLS]
        meta_dict[key] = vals
        
    return meta_dict, META_METRIC_COLS

def build_pitcher_cluster_dict(df):
    df_sorted = df.sort_values(by=["pitcher_id", "season", "game_month", "inning"], na_position='first')
    latest_pitcher_stats = df_sorted.drop_duplicates(subset=["pitcher_id"], keep="last").copy()
    
    features_for_clustering = [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        "asof_pitcher_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_strike_rate"
    ]
    
    X = latest_pitcher_stats[features_for_clustering].fillna(0.5).values
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    return dict(zip(latest_pitcher_stats["pitcher_id"].values, clusters))

def extract_zscore_stats(df):
    df_2024 = df[df["season"] == 2024]
    
    pitcher_rate_cols = [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"
    ]
    batter_rate_cols = [
        "asof_batter_success_rate", "asof_batter_middle_rate"
    ]
    
    C_pitcher, C_batter = 100.0, 50.0
    global_mean = 0.523
    
    stats_dict = {}
    
    for c in pitcher_rate_cols:
        if c in df_2024.columns:
            smoothed = (df_2024["asof_pitcher_n"] * df_2024[c].fillna(global_mean) + C_pitcher * global_mean) / (df_2024["asof_pitcher_n"] + C_pitcher)
            for g_type in ["R", "F"]:
                mask = df_2024["game_type"] == g_type
                if mask.any():
                    stats_dict[f"{c}_smoothed_mean_{g_type}"] = smoothed[mask].mean()
                    stats_dict[f"{c}_smoothed_std_{g_type}"] = smoothed[mask].std()
            
    for c in batter_rate_cols:
        if c in df_2024.columns:
            smoothed = (df_2024["asof_batter_n"] * df_2024[c].fillna(global_mean) + C_batter * global_mean) / (df_2024["asof_batter_n"] + C_batter)
            for g_type in ["R", "F"]:
                mask = df_2024["game_type"] == g_type
                if mask.any():
                    stats_dict[f"{c}_smoothed_mean_{g_type}"] = smoothed[mask].mean()
                    stats_dict[f"{c}_smoothed_std_{g_type}"] = smoothed[mask].std()
            
    return stats_dict

def engineer_features_v6(df, trackman_meta_dict=None, pitcher_cluster_dict=None, zscore_stats=None, trackman_stats_dict=None):
    df = df.copy()
    df["is_abs_era"] = (df["season"] >= 2024).astype(int)
    
    # Stadium owner team
    df["stadium_owner_team"] = np.where(df["top_bottom"] == 1, df["batter_team_id"], df["pitcher_team_id"])
    
    if trackman_meta_dict is not None:
        meta_arr = np.zeros((len(df), len(META_METRIC_COLS)), dtype=np.float32)
        seasons = df["season"].values.astype(int)
        p_hands = df["pitcher_hand"].map(HAND_MAP).fillna(2).values.astype(int)
        b_hands = df["batter_hand"].map(HAND_MAP).fillna(2).values.astype(int)
        default_vals = [0.0] * len(META_METRIC_COLS)
        for i in range(len(df)):
            s = seasons[i]
            if s > 2024: s = 2024
            k = (s, p_hands[i], b_hands[i])
            meta_arr[i] = trackman_meta_dict.get(k, default_vals)
        for idx, col in enumerate(META_METRIC_COLS):
            df[col] = meta_arr[:, idx]
            
    if pitcher_cluster_dict is not None:
        df["pitcher_style_cluster"] = df["pitcher_id"].map(pitcher_cluster_dict).fillna(-1).astype(int)
    else:
        df["pitcher_style_cluster"] = -1
        
    if trackman_stats_dict is not None:
        stats_map = trackman_stats_dict["stats"]
        global_means = trackman_stats_dict["global_means"]
        metrics = trackman_stats_dict["metrics"]
        
        stat_arr = np.zeros((len(df), len(metrics)), dtype=np.float32)
        pids = df["pitcher_id"].values
        seasons = df["season"].values.astype(int)
        
        for i in range(len(df)):
            s = seasons[i]
            # test.csv (2025) should look up s=2025 to include 2024 stats
            if s > 2025: s = 2025 
            k = (pids[i], s)
            stat_arr[i] = stats_map.get(k, global_means)
            
        for idx, col in enumerate(metrics):
            df[f"pitcher_hist_{col}"] = stat_arr[:, idx]
    
    df["count_state"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    df["is_hitter_count"] = ((df["balls_before"] >= 2) & (df["strikes_before"] <= 1)).astype(int)
    df["is_pitcher_count"] = ((df["strikes_before"] == 2) & (df["balls_before"] <= 1)).astype(int)
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["is_two_strike"] = (df["strikes_before"] == 2).astype(int)
    
    df["count_leverage"] = df["balls_before"] - df["strikes_before"]
    
    df["is_scoring_position"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
    df["is_bases_loaded"] = (df["num_runners_on"] == 3).astype(int)
    df["abs_score_diff"] = df["score_diff_pitcher_team"].abs()
    df["is_late_game"] = (df["inning"] >= 7).astype(int)
    df["is_clutch"] = (df["is_late_game"] & (df["abs_score_diff"] <= 2) & (df["li"] > 1.2)).astype(int)
    
    df["pitcher_count_combo"] = df["pitcher_id"].astype(str) + "_" + df["count_state"]
    df["batter_count_combo"] = df["batter_id"].astype(str) + "_" + df["count_state"]
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    
    C_pitcher, C_batter = 100.0, 50.0
    global_mean = 0.523
    
    pitcher_rate_cols = [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"
    ]
    for c in pitcher_rate_cols:
        if c in df.columns:
            smoothed_col = f"{c}_smoothed"
            df[smoothed_col] = (df["asof_pitcher_n"] * df[c].fillna(global_mean) + C_pitcher * global_mean) / (df["asof_pitcher_n"] + C_pitcher)
            
            if zscore_stats is not None:
                mean_2024_R = zscore_stats.get(f"{c}_smoothed_mean_R", 0.0)
                std_2024_R = zscore_stats.get(f"{c}_smoothed_std_R", 1.0)
                mean_2024_F = zscore_stats.get(f"{c}_smoothed_mean_F", mean_2024_R)
                std_2024_F = zscore_stats.get(f"{c}_smoothed_std_F", std_2024_R)
                
                # Apply appropriate mean/std based on game_type
                test_means = np.where(df["game_type"] == "R", mean_2024_R, mean_2024_F)
                test_stds = np.where(df["game_type"] == "R", std_2024_R, std_2024_F)
                
                df[f"{c}_zscore"] = (df[smoothed_col] - test_means) / (test_stds + 1e-5)
            else:
                df[f"{c}_zscore"] = df.groupby(["season", "game_type"])[smoothed_col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
        
    batter_rate_cols = ["asof_batter_success_rate", "asof_batter_middle_rate"]
    for c in batter_rate_cols:
        if c in df.columns:
            smoothed_col = f"{c}_smoothed"
            df[smoothed_col] = (df["asof_batter_n"] * df[c].fillna(global_mean) + C_batter * global_mean) / (df["asof_batter_n"] + C_batter)
            if zscore_stats is not None:
                mean_2024_R = zscore_stats.get(f"{c}_smoothed_mean_R", 0.0)
                std_2024_R = zscore_stats.get(f"{c}_smoothed_std_R", 1.0)
                mean_2024_F = zscore_stats.get(f"{c}_smoothed_mean_F", mean_2024_R)
                std_2024_F = zscore_stats.get(f"{c}_smoothed_std_F", std_2024_R)
                
                test_means = np.where(df["game_type"] == "R", mean_2024_R, mean_2024_F)
                test_stds = np.where(df["game_type"] == "R", std_2024_R, std_2024_F)
                
                df[f"{c}_zscore"] = (df[smoothed_col] - test_means) / (test_stds + 1e-5)
            else:
                df[f"{c}_zscore"] = df.groupby(["season", "game_type"])[smoothed_col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))

    df["diff_prev1_success"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate_smoothed"]
    df["diff_prev3_success"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate_smoothed"]
    df["diff_prev5_success"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate_smoothed"]
    
    df["diff_prev1_middle"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_middle_rate_smoothed"]
    df["diff_prev3_middle"] = df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_middle_rate_smoothed"]
    df["diff_prev5_middle"] = df["asof_pitcher_prev5_game_middle_rate"] - df["asof_pitcher_middle_rate_smoothed"]
    
    df["pitcher_volatility"] = df["diff_prev1_success"].abs() + df["diff_prev3_success"].abs()
    df["li_log"] = np.log1p(df["li"].clip(lower=0))
    df["pressure_interaction"] = df["pitcher_volatility"] * df["li_log"]
    
    df["strike_to_ball_ratio"] = df["asof_pitcher_strike_rate_smoothed"] / (df["asof_pitcher_ball_rate_smoothed"] + 1e-5)
    df["reverse_to_success_ratio"] = df["asof_pitcher_reverse_rate_smoothed"] / (df["asof_pitcher_success_rate_smoothed"] + 1e-5)
    df["middle_to_success_ratio"] = df["asof_pitcher_middle_rate_smoothed"] / (df["asof_pitcher_success_rate_smoothed"] + 1e-5)
    
    p_fast = df["asof_pitcher_fastball_rate"].clip(1e-5, 1.0)
    p_break = df["asof_pitcher_breaking_rate"].clip(1e-5, 1.0)
    p_off = df["asof_pitcher_offspeed_rate"].clip(1e-5, 1.0)
    df["pitchmix_entropy"] = - (p_fast * np.log(p_fast) + p_break * np.log(p_break) + p_off * np.log(p_off))
    
    df["win_expectancy_diff"] = df["home_win_expectancy"] - df["away_win_expectancy"]
    
    return df
