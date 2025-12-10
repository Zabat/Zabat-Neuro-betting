#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurocomputational Modeling of Cognitive Biases in Sports Betting
==================================================================

A deep learning framework for modeling and predicting bias-driven betting behavior 
using LSTM networks with temporal attention mechanisms.

This implementation models three major cognitive biases:
    - Hot-Hand Fallacy
    - Loss Chasing
    - Confirmation Bias

Authors: René Manassé Galekwa, Selain K. Kasereka, Kyandoghere Kyamakya
Institution: University of Klagenfurt / University of Kinshasa
"""

import os
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SEEDS: List[int] = [42, 43, 44]
VERBOSE: int = 2


# =============================================================================
# Bias Parameters
# =============================================================================

@dataclass
class BiasParams:
    """Configuration parameters for cognitive bias simulation.
    
    Attributes:
        hot_hand_boost: Probability boost for high-risk bet after win streak
        hot_hand_streak_k: Number of consecutive wins to trigger hot-hand effect
        loss_chase_boost: Probability boost for high-risk bet after loss streak
        loss_chase_streak_k: Number of consecutive losses to trigger loss chasing
        confirm_stickiness: Persistence coefficient for confirmation bias
        base_highrisk_p: Base probability of choosing high-risk bet
        base_win_p_low: Base win probability for low-risk bets
        base_win_p_high: Base win probability for high-risk bets
    """
    hot_hand_boost: float = 0.35
    hot_hand_streak_k: int = 3
    loss_chase_boost: float = 0.40
    loss_chase_streak_k: int = 2
    confirm_stickiness: float = 0.25
    base_highrisk_p: float = 0.42
    base_win_p_low: float = 0.62
    base_win_p_high: float = 0.33


# =============================================================================
# Utility Functions
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Data Generation
# =============================================================================

def simulate_bettor_sequence(
    T: int,
    bias_type: str,
    params: BiasParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Simulate a single bettor's betting sequence with cognitive biases.
    
    Args:
        T: Number of betting rounds
        bias_type: Type of cognitive bias ('hot_hand', 'loss_chasing', 
                   'confirmation', or 'rational')
        params: Bias configuration parameters
    
    Returns:
        Tuple containing:
            - a: Array of actions (0=low-risk, 1=high-risk)
            - o: Array of outcomes (0=loss, 1=win)
            - p_env: Array of win probabilities [p_high, p_low] per step
            - info: Dictionary with metadata
    """
    a = np.zeros(T, dtype=int)
    o = np.zeros(T, dtype=int)
    
    # Generate environment win probabilities with noise
    p_high = np.clip(np.random.normal(params.base_win_p_high, 0.03, size=T), 0.05, 0.95)
    p_low = np.clip(np.random.normal(params.base_win_p_low, 0.03, size=T), 0.05, 0.95)
    
    # Initialize bettor state
    preferred_action = np.random.choice([0, 1])  # For confirmation bias
    win_streak = 0
    loss_streak = 0
    confirm_stickiness = params.confirm_stickiness
    
    for t in range(T):
        p_choose_high = params.base_highrisk_p
        
        # Apply hot-hand fallacy
        if bias_type == "hot_hand" and win_streak >= params.hot_hand_streak_k:
            p_choose_high = min(1.0, p_choose_high + params.hot_hand_boost)
        
        # Apply loss chasing
        if bias_type == "loss_chasing" and loss_streak >= params.loss_chase_streak_k:
            p_choose_high = min(1.0, p_choose_high + params.loss_chase_boost)
        
        # Apply confirmation bias
        if bias_type == "confirmation":
            if preferred_action == 1:
                p_choose_high = min(1.0, p_choose_high + confirm_stickiness)
            else:
                p_choose_high = max(0.0, p_choose_high - confirm_stickiness)
        
        # Small rational nudge toward higher expected value
        if p_high[t] > p_low[t]:
            p_choose_high += 0.05
        else:
            p_choose_high -= 0.05
        
        p_choose_high = float(np.clip(p_choose_high, 0.0, 1.0))
        
        # Generate action and outcome
        a[t] = np.random.rand() < p_choose_high
        o[t] = np.random.rand() < (p_high[t] if a[t] == 1 else p_low[t])
        
        # Update streaks
        if o[t] == 1:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
        
        # Adapt stickiness for confirmation bias
        if bias_type == "confirmation":
            if a[t] == preferred_action and o[t] == 1:
                confirm_stickiness = min(0.40, confirm_stickiness + 0.01)
            elif a[t] != preferred_action and o[t] == 0:
                confirm_stickiness = min(0.40, confirm_stickiness + 0.005)
    
    info = {"preferred_action": preferred_action}
    return a, o, np.stack([p_high, p_low], axis=1), info


def build_per_step_features(a: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Build feature vectors for each time step.
    
    Features per step:
        - a_t: Action taken (0 or 1)
        - o_t: Outcome received (0 or 1)
        - win_streak_norm: Normalized win streak length [0, 1]
        - loss_streak_norm: Normalized loss streak length [0, 1]
        - delta3: Normalized sum of last 3 outcomes [-1, 1]
    
    Args:
        a: Array of actions
        o: Array of outcomes
    
    Returns:
        Feature matrix of shape (T, 5)
    """
    T = len(a)
    win_stk = np.zeros(T, dtype=int)
    loss_stk = np.zeros(T, dtype=int)
    
    w, l = 0, 0
    for t in range(T):
        if o[t] == 1:
            w += 1
            l = 0
        else:
            l += 1
            w = 0
        win_stk[t] = min(w, 5)
        loss_stk[t] = min(l, 5)
    
    # Compute delta3: rolling sum of last 3 outcomes
    delta3 = np.zeros(T, dtype=float)
    for t in range(T):
        s = sum(o[max(0, t-2):t+1])
        delta3[t] = (s - 1.5) / 1.5  # Normalize to [-1, 1]
    
    # Normalize streaks to [0, 1]
    win_norm = win_stk / 5.0
    loss_norm = loss_stk / 5.0
    
    feats = np.stack([
        a.astype(float),
        o.astype(float),
        win_norm,
        loss_norm,
        delta3
    ], axis=1)
    
    return feats


def generate_synthetic_dataset(
    seed: int,
    N_bettors: int = 1000,
    T: int = 100,
    bias_mix: Dict[str, float] = None,
    window_k: int = 15,
    params: BiasParams = None
) -> Dict:
    """Generate synthetic dataset of biased betting sequences.
    
    Args:
        seed: Random seed for reproducibility
        N_bettors: Number of virtual bettors
        T: Number of betting rounds per bettor
        bias_mix: Dictionary mapping bias types to proportions
        window_k: Sliding window size for sequences
        params: Bias configuration parameters
    
    Returns:
        Dictionary containing train/val/test splits and metadata
    """
    set_all_seeds(seed)
    
    if bias_mix is None:
        bias_mix = {
            "hot_hand": 0.3,
            "loss_chasing": 0.3,
            "confirmation": 0.2,
            "rational": 0.2
        }
    if params is None:
        params = BiasParams()
    
    all_X, all_y, all_flat, all_ev, groups = [], [], [], [], []
    bettors_meta = []
    
    bias_types = np.random.choice(
        list(bias_mix.keys()),
        size=N_bettors,
        p=list(bias_mix.values())
    )
    
    for i in range(N_bettors):
        bias_type = bias_types[i]
        a, o, p_env, info = simulate_bettor_sequence(T, bias_type, BiasParams(**vars(params)))
        feats = build_per_step_features(a, o)
        
        for t in range(window_k, T):
            x_window = feats[t-window_k:t, :]
            y_next = a[t]
            
            all_X.append(x_window)
            all_y.append(y_next)
            all_flat.append(x_window.flatten())
            ev_proxy = p_env[t, 0] - p_env[t, 1]
            all_ev.append(ev_proxy)
            groups.append(i)
        
        bettors_meta.append({"bias": bias_type, "info": info})
    
    # Convert to arrays
    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int32)
    X_flat = np.asarray(all_flat, dtype=np.float32)
    ev = np.asarray(all_ev, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int32)
    
    # Group-wise split (bettor-wise to prevent leakage)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    
    n_train = int(0.7 * len(unique_groups))
    n_val = int(0.15 * len(unique_groups))
    
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train+n_val])
    test_groups = set(unique_groups[n_train+n_val:])
    
    def mask_for(gs):
        return np.isin(groups, list(gs))
    
    return {
        "X_train": X[mask_for(train_groups)],
        "y_train": y[mask_for(train_groups)],
        "X_val": X[mask_for(val_groups)],
        "y_val": y[mask_for(val_groups)],
        "X_test": X[mask_for(test_groups)],
        "y_test": y[mask_for(test_groups)],
        "X_flat_train": X_flat[mask_for(train_groups)],
        "X_flat_val": X_flat[mask_for(val_groups)],
        "X_flat_test": X_flat[mask_for(test_groups)],
        "ev_train": ev[mask_for(train_groups)],
        "ev_val": ev[mask_for(val_groups)],
        "ev_test": ev[mask_for(test_groups)],
        "meta": {
            "bettors": bettors_meta,
            "window_k": window_k,
            "seed": seed
        }
    }


# =============================================================================
# Model Architecture
# =============================================================================

class TemporalAttention(layers.Layer):
    """Content-based temporal attention mechanism.
    
    Computes attention weights over LSTM hidden states to focus on
    the most relevant time steps for prediction.
    
    Args:
        hidden_units: Dimension of the attention hidden layer
    """
    
    def __init__(self, hidden_units: int):
        super().__init__()
        self.W_h = layers.Dense(hidden_units, use_bias=False)
        self.W_s = layers.Dense(hidden_units, use_bias=False)
        self.v = layers.Dense(1, use_bias=False)
    
    def call(self, H, mask=None):
        """Compute attention-weighted context vector.
        
        Args:
            H: LSTM hidden states of shape (batch, time, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Use last hidden state as query
        s = H[:, -1, :]
        s_exp = tf.expand_dims(s, axis=1)
        s_exp = tf.repeat(s_exp, repeats=tf.shape(H)[1], axis=1)
        
        # Compute attention scores
        e = self.v(tf.nn.tanh(self.W_h(H) + self.W_s(s_exp)))
        e = tf.squeeze(e, axis=-1)
        
        # Normalize with softmax
        alpha = tf.nn.softmax(e, axis=1)
        
        # Compute weighted context
        context = tf.reduce_sum(tf.expand_dims(alpha, -1) * H, axis=1)
        
        return context, alpha


def build_lstm_attention_model(
    input_timesteps: int,
    feature_dim: int,
    lstm_units: int = 96,
    dropout: float = 0.15
) -> Tuple[keras.Model, keras.Model]:
    """Build LSTM model with temporal attention mechanism.
    
    Args:
        input_timesteps: Number of time steps in input sequence
        feature_dim: Number of features per time step
        lstm_units: Number of LSTM hidden units
        dropout: Dropout rate
    
    Returns:
        Tuple of (main_model, attention_model) for prediction and interpretation
    """
    inp = keras.Input(shape=(input_timesteps, feature_dim))
    
    # LSTM encoding
    x = layers.LSTM(lstm_units, return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    
    # Attention mechanism
    context, attn = TemporalAttention(lstm_units)(x)
    
    # Combine context with last hidden state
    last_h = x[:, -1, :]
    z = layers.Concatenate()([context, last_h])
    
    # Output layers
    z = layers.Dense(128, activation="relu")(z)
    out = layers.Dense(1, activation="sigmoid")(z)
    
    # Main model for prediction
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")]
    )
    
    # Attention model for interpretability
    attn_model = keras.Model(inputs=inp, outputs=attn)
    
    return model, attn_model


# =============================================================================
# Baseline Models
# =============================================================================

def rational_agent_probs(ev_proxy: np.ndarray, scale: float = 5.0) -> np.ndarray:
    """Compute rational agent probabilities based on expected value.
    
    Maps EV proxy to [0, 1] via logistic function.
    
    Args:
        ev_proxy: Expected value difference (p_high - p_low)
        scale: Logistic function scaling factor
    
    Returns:
        Probability estimates
    """
    return 1.0 / (1.0 + np.exp(-scale * ev_proxy))


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr: float,
    name: str
) -> Tuple[float, float, float]:
    """Evaluate model performance at a given threshold.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        thr: Classification threshold
        name: Model name for display
    
    Returns:
        Tuple of (accuracy, f1_score, auc)
    """
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"[{name}] thr={thr:.2f}  ACC={acc*100:.1f}%  F1={f1:.3f}  AUC={auc:.3f}")
    
    return acc, f1, auc


def tune_threshold_by_f1(
    y_val: np.ndarray,
    p_val: np.ndarray,
    lo: float = 0.3,
    hi: float = 0.7,
    steps: int = 81
) -> Tuple[float, float]:
    """Find optimal threshold that maximizes F1-score.
    
    Args:
        y_val: Validation labels
        p_val: Validation predictions
        lo: Lower threshold bound
        hi: Upper threshold bound
        steps: Number of thresholds to evaluate
    
    Returns:
        Tuple of (best_threshold, best_f1_score)
    """
    thresholds = np.linspace(lo, hi, steps)
    best_thr, best_f1 = 0.5, -1.0
    
    for t in thresholds:
        f1 = f1_score(y_val, (p_val >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    
    return best_thr, best_f1


# =============================================================================
# Main Experiment
# =============================================================================

def run_once(
    seed: int,
    n_bettors: int = 1000,
    T: int = 100,
    window_k: int = 15
) -> Dict:
    """Run a single experiment with the given configuration.
    
    Args:
        seed: Random seed
        n_bettors: Number of virtual bettors
        T: Betting rounds per bettor
        window_k: Sequence window size
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with seed={seed}")
    print(f"{'='*60}")
    
    # Generate data
    ds = generate_synthetic_dataset(
        seed=seed,
        N_bettors=n_bettors,
        T=T,
        window_k=window_k,
        bias_mix={
            "hot_hand": 0.3,
            "loss_chasing": 0.3,
            "confirmation": 0.2,
            "rational": 0.2
        },
        params=BiasParams()
    )
    
    Xtr, ytr = ds["X_train"], ds["y_train"]
    Xva, yva = ds["X_val"], ds["y_val"]
    Xte, yte = ds["X_test"], ds["y_test"]
    
    # Build and train model
    model, attn_model = build_lstm_attention_model(
        input_timesteps=Xtr.shape[1],
        feature_dim=Xtr.shape[2],
        lstm_units=96,
        dropout=0.15
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]
    
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=120,
        batch_size=512,
        verbose=VERBOSE,
        callbacks=callbacks
    )
    
    # Generate predictions
    p_val = model.predict(Xva, batch_size=1024, verbose=0).ravel()
    p_te = model.predict(Xte, batch_size=1024, verbose=0).ravel()
    
    # Tune threshold on validation set
    best_thr, best_val_f1 = tune_threshold_by_f1(yva, p_val, lo=0.3, hi=0.7, steps=81)
    print(f"\nBest validation F1: {best_val_f1:.3f} at threshold={best_thr:.2f}")
    
    # Evaluate LSTM-Attention
    print("\n--- Test Set Evaluation ---")
    acc05, f105, auc05 = evaluate_at_threshold(yte, p_te, 0.5, "LSTM+Attn (θ=0.5)")
    acct, f1t, auct = evaluate_at_threshold(yte, p_te, best_thr, "LSTM+Attn (tuned)")
    
    # Evaluate baselines
    print("\n--- Baseline Comparison ---")
    
    # Rational Agent
    p_ra = rational_agent_probs(ds["ev_test"], scale=5.0)
    evaluate_at_threshold(yte, p_ra, 0.5, "Rational Agent")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=200, solver="lbfgs")
    lr.fit(ds["X_flat_train"], ytr)
    p_lr = lr.predict_proba(ds["X_flat_test"])[:, 1]
    evaluate_at_threshold(yte, p_lr, 0.5, "Logistic Regression")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=seed, n_jobs=-1)
    rf.fit(ds["X_flat_train"], ytr)
    p_rf = rf.predict_proba(ds["X_flat_test"])[:, 1]
    evaluate_at_threshold(yte, p_rf, 0.5, "Random Forest")
    
    # Attention analysis
    print("\n--- Attention Analysis ---")
    idx = np.random.randint(0, Xte.shape[0])
    aw = attn_model.predict(Xte[idx:idx+1], verbose=0).ravel()
    
    print("Example attention weights (older → newer):")
    print(np.round(aw, 3))
    print("\nWindow features [action, outcome, win_streak, loss_streak, delta3]:")
    print(np.round(Xte[idx], 3))
    print(f"\nTrue label: {int(yte[idx])}, Predicted P(high-risk): {float(p_te[idx]):.3f}")
    
    return {
        "seed": seed,
        "acc05": acc05, "f105": f105, "auc05": auc05,
        "acct": acct, "f1t": f1t, "auct": auct
    }


def summarize_runs(results: List[Dict]) -> None:
    """Print summary statistics across multiple experiment runs.
    
    Args:
        results: List of result dictionaries from run_once()
    """
    def aggregate(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)
    
    print("\n" + "="*60)
    print("Summary Statistics Across Seeds")
    print("="*60)
    
    for label in ["acc05", "f105", "auc05", "acct", "f1t", "auct"]:
        mean, std = aggregate(label)
        if label.startswith("acc"):
            print(f"{label}: {mean*100:.2f} ± {std*100:.2f}%")
        else:
            print(f"{label}: {mean:.3f} ± {std:.3f}")


def main():
    """Main entry point for the experiment."""
    results = []
    
    for seed in DEFAULT_SEEDS:
        set_all_seeds(seed)
        res = run_once(seed=seed, n_bettors=600, T=100, window_k=15)
        results.append(res)
    
    summarize_runs(results)


if __name__ == "__main__":
    main()
