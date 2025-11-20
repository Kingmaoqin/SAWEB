import numpy as np
import pandas as pd
import torchtuples as tt
from sklearn.model_selection import train_test_split

from utils.pycox_setup import ensure_pycox_writable

ensure_pycox_writable()

from pycox.evaluation import EvalSurv
from pycox.models import CoxPH

def run_deepsurv(data, config):
    """
    Run DeepSurv (CoxPH) on data.
    Data must have columns: features, "duration", "event".
    config: dict with keys "batch_size", "epochs", "lr".
    Returns a dict with evaluation metrics and predicted survival curves.
    Keys include:
      - "Partial Log Likelihood"
      - "C-index (Train)"
      - "C-index (Validation)"
      - "C-index (Test)"
      - "Integrated Brier Score"
      - "Integrated Negative Log-Likelihood"
      - "Surv_Test": predicted survival curves DataFrame.
    """
    # Reset index并检查必要列
    data = data.reset_index(drop=True)
    required_cols = {"duration", "event"}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    X = data.drop(columns=required_cols)
    non_numeric = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_time = data["duration"]
    y_event = data["event"]
    
    # 分层拆分数据集
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X, y_time, y_event,
        test_size=0.2,
        stratify=y_event,
        random_state=42
    )
    
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
    
    # 将train_df再拆分为训练集和验证集
    train_idx, val_idx = train_test_split(
        train_df.index,
        test_size=0.2,
        stratify=train_df["event"],
        random_state=42
    )
    train_data = train_df.loc[train_idx]
    val_data = train_df.loc[val_idx]
    
    # 准备特征，转换为float32
    def prepare_features(df):
        features = df.drop(columns=required_cols)
        return features.astype("float32").values
    x_train = prepare_features(train_data)
    x_val = prepare_features(val_data)
    x_test = prepare_features(test_df)
    
    # 准备目标变量
    def prepare_targets(df):
        durations = df["duration"].astype("float32").values
        events = df["event"].astype("float32").values
        return (durations, events)
    y_train = prepare_targets(train_data)
    y_val = prepare_targets(val_data)
    test_durations = test_df["duration"].astype("float32").values
    test_events = test_df["event"].astype("float32").values
    
    # 模型配置
    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features=1,
        batch_norm=True, dropout=0.1
    )
    model = CoxPH(net, tt.optim.Adam)
    
    batch_size = config.get("batch_size", 256)
    epochs = config.get("epochs", 64)
    model.optimizer.set_lr(config.get("lr", 0.1))
    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    verbose = False
    
    model.fit(
        x_train, y_train,
        batch_size, epochs, callbacks, verbose,
        val_data=(x_val, y_val), val_batch_size=batch_size
    )
    
    # 计算部分对数似然（仅用于监控，可选）
    partial_ll = model.partial_log_likelihood(x_val, y_val).mean()
    
    model.compute_baseline_hazards()
    
    surv_train = model.predict_surv_df(x_train)
    surv_val = model.predict_surv_df(x_val)
    surv_test = model.predict_surv_df(x_test)
    
    ev_train = EvalSurv(surv_train, y_train[0], y_train[1], censor_surv="km")
    c_index_train = ev_train.concordance_td()
    ev_val = EvalSurv(surv_val, y_val[0], y_val[1], censor_surv="km")
    c_index_val = ev_val.concordance_td()
    ev_test = EvalSurv(surv_test, test_durations, test_events, censor_surv="km")
    c_index_test = ev_test.concordance_td()
    time_grid = np.linspace(test_durations.min(), test_durations.max(), 100)
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
