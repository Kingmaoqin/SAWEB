import numpy as np
import pandas as pd
import torchtuples as tt
from sklearn.model_selection import train_test_split

from utils.pycox_setup import ensure_pycox_writable

ensure_pycox_writable()

from pycox.evaluation import EvalSurv
from pycox.models import DeepHitSingle

def run_deephit(data, config):
    """
    Run DeepHit on data.
    Data must have columns: features, "duration", "event".
    config: dict with keys "batch_size", "epochs", "lr", "num_intervals", "out_features".
    Returns a dict with evaluation metrics and predicted survival curves.
    Keys include:
      - "C-index (Train)"
      - "C-index (Validation)"
      - "C-index (Test)"
      - "Integrated Brier Score"
      - "Integrated Negative Log-Likelihood"
      - "Surv_Test": predicted survival curves DataFrame.
    """
    # Reset index，确保数据对齐
    data = data.reset_index(drop=True)
    required_cols = {"duration", "event"}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Split features and target
    X = data.drop(columns=required_cols)
    non_numeric = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_time = data["duration"]
    y_event = data["event"]
    
    # Stratified split以保持event比例
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
    )
    
    # 重建DataFrame确保索引连续
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
    
    # 进一步将train_df拆分为训练集和验证集
    train_idx, val_idx = train_test_split(
        train_df.index, test_size=0.2, stratify=train_df["event"], random_state=42
    )
    train_data = train_df.loc[train_idx]
    val_data = train_df.loc[val_idx]
    
    # 准备特征（保证类型为float32）
    def prepare_features(df):
        features = df.drop(columns=required_cols)
        assert len(features) == len(df), "Feature dimension mismatch"
        return features.astype("float32").values
    
    x_train = prepare_features(train_data)
    x_val = prepare_features(val_data)
    x_test = prepare_features(test_df)
    
    # 对于DeepHit，需要离散化生存时间
    num_intervals = config.get("num_intervals", 20)
    time_bins = np.linspace(data["duration"].min(), data["duration"].max(), num_intervals)
    
    def discretize(df):
        return np.digitize(df["duration"], bins=time_bins) - 1
    
    y_train_discrete = discretize(train_data)
    y_val_discrete = discretize(val_data)
    y_test_discrete = discretize(test_df)
    
    # 目标转为tuple形式： (离散时间, event)
    y_train_tuple = (y_train_discrete.astype("int64"), train_data["event"].astype("float32").values)
    y_val_tuple = (y_val_discrete.astype("int64"), val_data["event"].astype("float32").values)
    durations_test = test_df["duration"].astype("float32").values
    events_test = test_df["event"].astype("float32").values

    # 模型配置
    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = config.get("out_features", 4000)
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features,
        batch_norm=True, dropout=0.01
    )
    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1)

    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 512)
    model.optimizer.set_lr(config.get("lr", 0.1))
    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    verbose = False

    model.fit(
        x_train, y_train_tuple,
        batch_size, epochs, callbacks, verbose,
        val_data=(x_val, y_val_tuple), val_batch_size=batch_size
    )

    # 预测生存曲线
    surv_train = model.predict_surv_df(x_train)
    surv_val = model.predict_surv_df(x_val)
    surv_test = model.predict_surv_df(x_test)
    
    # 计算评价指标
    ev_train = EvalSurv(surv_train, y_train_tuple[0], y_train_tuple[1], censor_surv="km")
    c_index_train = ev_train.concordance_td()
    ev_val = EvalSurv(surv_val, y_val_tuple[0], y_val_tuple[1], censor_surv="km")
    c_index_val = ev_val.concordance_td()
    ev_test = EvalSurv(surv_test, y_test_discrete, events_test, censor_surv="km")
    c_index_test = ev_test.concordance_td()
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    integrated_brier_score = ev_test.integrated_brier_score(time_grid)
    integrated_nbll = ev_test.integrated_nbll(time_grid)
    
    return {
        "C-index (Train)": c_index_train,
        "C-index (Validation)": c_index_val,
        "C-index (Test)": c_index_test,
        "Integrated Brier Score": integrated_brier_score,
        "Integrated Negative Log-Likelihood": integrated_nbll,
        "Surv_Test": surv_test
    }
