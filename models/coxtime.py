import numpy as np
import pandas as pd
import torchtuples as tt
from sklearn.model_selection import train_test_split

from utils.pycox_setup import ensure_pycox_writable

ensure_pycox_writable()

from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

def run_coxtime(data, config):
    """
    Robust CoxTime implementation with dimension consistency checks.

    Args:
        data: DataFrame containing feature columns, 'duration' and 'event' columns.
        config: Dictionary with hyperparameters, e.g., {"batch_size": 64, "epochs": 512, "lr": 0.01}.
    
    Returns:
        Dictionary with evaluation metrics and predicted survival curves.
        Keys include:
          - "Partial Log Likelihood"
          - "C-index (Train)"
          - "C-index (Validation)"
          - "C-index (Test)"
          - "Integrated Brier Score"
          - "Integrated Negative Log-Likelihood"
          - "Surv_Test": predicted survival curves DataFrame.
    """
    # Reset index to prevent alignment issues
    data = data.reset_index(drop=True)
    
    # Validate required columns
    required_cols = {"duration", "event"}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Split data: X features and y target
    X = data.drop(columns=required_cols)

    # Encode non-numeric columns globally before splitting so train/val/test stay aligned
    non_numeric = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=False)

    # Ensure all values are numeric floats (booleans cast cleanly) and fill NaNs from coercion
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_time = data["duration"]
    y_event = data["event"]
    
    # Stratified split to maintain event distribution
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X, y_time, y_event,
        test_size=0.2,
        stratify=y_event,
        random_state=42
    )
    
    # Recombine to DataFrames，保证索引一致
    train_df = pd.concat([
        X_train.reset_index(drop=True),
        y_time_train.reset_index(drop=True),
        y_event_train.reset_index(drop=True)
    ], axis=1)
    test_df = pd.concat([
        X_test.reset_index(drop=True),
        y_time_test.reset_index(drop=True),
        y_event_test.reset_index(drop=True)
    ], axis=1)
    
    # Further split training into train/validation
    train_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.2,
        stratify=train_df["event"],
        random_state=42
    )
    train_data = train_df.loc[train_idx]
    val_data = train_df.loc[val_idx]
    
    # Prepare features as numpy arrays with consistency check
    def prepare_features(df):
        features = df.drop(columns=required_cols)
        assert len(features) == len(df), "Feature dimension mismatch"
        return features.astype("float32").values
    
    x_train = prepare_features(train_data)
    x_val = prepare_features(val_data)
    x_test = prepare_features(test_df)
    
    # Prepare target variables
    def prepare_targets(df):
        durations = df["duration"].astype("float32").values
        events = df["event"].astype("float32").values
        assert len(durations) == len(events) == len(df), "Target dimension mismatch"
        return (durations, events)
    
    y_train = prepare_targets(train_data)
    y_val = prepare_targets(val_data)
    
    # Model configuration
    in_features = x_train.shape[1]
    net = MLPVanillaCoxTime(
        in_features=in_features,
        num_nodes=[32, 32],
        batch_norm=True,
        dropout=0.1
    )
    model = CoxTime(net, tt.optim.Adam)
    
    # Training configuration
    model.optimizer.set_lr(config.get("lr", 0.01))
    model.fit(
        x_train, y_train,
        batch_size=config.get("batch_size", 64),
        epochs=config.get("epochs", 512),
        val_data=(x_val, y_val),
        callbacks=[tt.callbacks.EarlyStopping(patience=10)],
        verbose=False
    )
    
    # Compute baseline hazards
    model.compute_baseline_hazards()
    
    # Predict survival curves
    surv_train = model.predict_surv_df(x_train)
    surv_val = model.predict_surv_df(x_val)
    surv_test = model.predict_surv_df(x_test)
    # 检查预测结果维度是否与测试样本数量一致
    assert surv_test.shape[1] == x_test.shape[0], \
        f"Prediction dimension mismatch: {surv_test.shape[1]} vs {x_test.shape[0]}"
    
    # Compute evaluation metrics using test set targets
    test_durations = test_df["duration"].astype("float32").values
    test_events = test_df["event"].astype(bool).values 
    ev_test = EvalSurv(
        surv=surv_test,
        durations=test_durations,
        events=test_events,
        censor_surv="km"
    )
    
    # Create safe time grid
    time_grid = np.linspace(
        start=max(1, test_durations.min()),  
        stop=test_durations.max(),
        num=100
    )
    
    # Calculate metrics
    partial_ll = model.partial_log_likelihood(x_val, y_val).mean()
    c_index_train = EvalSurv(surv_train, *y_train, censor_surv="km").concordance_td()
    c_index_val = EvalSurv(surv_val, *y_val, censor_surv="km").concordance_td()
    c_index_test = ev_test.concordance_td()
    integrated_brier_score = ev_test.integrated_brier_score(time_grid)
    integrated_nbll = ev_test.integrated_nbll(time_grid)
    
    return {
        "Partial Log Likelihood": partial_ll,
        "C-index (Train)": c_index_train,
        "C-index (Validation)": c_index_val,
        "C-index (Test)": c_index_test,
        "Integrated Brier Score": integrated_brier_score,
        "Integrated Negative Log-Likelihood": integrated_nbll,
        "Surv_Test": surv_test
    }
