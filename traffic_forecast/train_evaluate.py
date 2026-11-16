"""
=============================================================================
train_evaluate.py  —  Train cục bộ & đánh giá từng phương pháp
=============================================================================
Đồ án: Mô phỏng kiến trúc Node-Agent sử dụng Học Liên Bang

VAI TRÒ: Điều phối toàn bộ pipeline train + đánh giá cho 1 phương pháp
  × 1 khung giờ. Là cầu nối giữa data_loader, models và federated.

Chức năng chính:
  train_local_all()      — Train tất cả nodes cục bộ (bước 1 FL)
  run_one_method()       — Pipeline đầy đủ: local → FL → metrics → save
  save_node_models()     — Lưu weights tất cả nodes ra file .npz
  load_node_models()     — Nạp weights đã lưu, bỏ qua bước train

Cơ chế lưu/nạp model:
  Sau train + FL, weights từng node được lưu vào:
    saved_models/<session>/<method>/<segment_hash>.npz
  Lần chạy sau, nếu file đã tồn tại → nạp ngay, không train lại.
  Dùng --force-retrain để ép train lại từ đầu.
=============================================================================
"""

import hashlib
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config as _cfg   # đọc LOCAL_EPOCHS/FL_ROUNDS... lúc chạy (để --quick có tác dụng)
from config import (
    SEQ_LEN, TRAIN_RATIO, LOCAL_EPOCHS, LEARNING_RATE,
    FL_ROUNDS, FL_LOCAL_EPOCHS, RANDOM_SEED,
    model_save_dir,
)
from data_loader import (
    make_sequences, normalize, compute_metrics,
    get_node_series, get_node_day_series, build_train_test,
)
from models import (
    GRUModel, LSTMModel, BiLSTMModel,
    create_model, METHOD_NAMES, HAS_TORCH,
)
from federated import run_federated, _train_model, fedavg_weights

np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass


# =============================================================================
# HELPER: TÊN FILE LƯU MODEL
# =============================================================================

def _seg_filename(seg: str) -> str:
    """
    Chuyển tên segment thành tên file an toàn (hash MD5 8 ký tự).
    Tránh ký tự đặc biệt trong tên đường phố.
    """
    return hashlib.md5(seg.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# 1. LƯU / NẠP WEIGHTS TOÀN BỘ NODE
# =============================================================================

def save_node_models(node_data: Dict[str, dict],
                     session_key: str, method: str,
                     run_info: Optional[dict] = None) -> int:
    """
    Lưu weights của tất cả nodes sau khi train + FL.

    Mỗi node → 1 file .npz trong saved_models/<session>/<method>/
    Cũng lưu metadata (mn, mx) để denormalize khi predict.

    Trả về: số file đã lưu thành công.
    """
    save_dir = model_save_dir(session_key, method)
    count = 0
    for seg, d in node_data.items():
        fname = _seg_filename(seg)
        path  = os.path.join(save_dir, fname)
        # Lưu weights model
        d["model"].save(path)
        # Lưu metadata normalize
        meta_path = path + "_meta.npz"
        np.savez(meta_path, mn=np.array([d["mn"]]), mx=np.array([d["mx"]]),
                 seg=np.array([seg]))
        count += 1
    # Lưu bảng mapping seg → fname
    map_path = os.path.join(save_dir, "_seg_map.npz")
    segs  = list(node_data.keys())
    fnames = [_seg_filename(s) for s in segs]
    np.savez(map_path, segs=np.array(segs), fnames=np.array(fnames))
    # Lưu kết quả đánh giá (local trước FL, FL, log các vòng) để lần sau
    # nạp cache mà không phải train/FL lại
    if run_info is not None:
        with open(os.path.join(save_dir, "_run_info.json"), "w",
                  encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2, default=float)
    print(f"        [SAVE] Đã lưu {count} model vào {save_dir}/")
    return count


def build_run_info(method: str, node_data: Dict[str, dict],
                   fl_metrics: List[dict], round_log: List[dict]) -> dict:
    """Gói kết quả 1 lần train+FL để lưu cạnh model (xem save_node_models)."""
    return {
        "fl_rounds": len(round_log),
        "local_metrics": [{
            "method": method, "segment": seg, "road": d["road"],
            "MAE_local": d["local_mae"], "RMSE_local": d["local_rmse"],
            "MAPE_local": d["local_mape"],
            "n_train": d["train_size"], "n_test": d["test_size"],
        } for seg, d in node_data.items()],
        "fl_metrics": fl_metrics,
        "round_log": round_log,
    }


def load_run_info(session_key: str, method: str) -> Optional[dict]:
    """Đọc _run_info.json đã lưu; None nếu chưa có hoặc hỏng."""
    path = os.path.join(model_save_dir(session_key, method), "_run_info.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_node_models(session_key: str, method: str,
                     records: list) -> Optional[Dict[str, dict]]:
    """
    Nạp weights đã lưu. Trả về None nếu chưa có file.

    Dùng trước bước train để kiểm tra xem đã có model chưa:
        node_data = load_node_models(session_key, method, records)
        if node_data is None:
            node_data = train_local_all(...)
            save_node_models(node_data, session_key, method)
    """
    save_dir = model_save_dir(session_key, method)
    map_path = os.path.join(save_dir, "_seg_map.npz")
    if not os.path.exists(map_path):
        return None

    data = np.load(map_path, allow_pickle=True)
    segs   = data["segs"].tolist()
    fnames = data["fnames"].tolist()

    # Rebuild node_series để lấy X_te, y_te cho evaluate
    node_runs = get_node_day_series(records)
    result: Dict[str, dict] = {}

    for seg, fname in zip(segs, fnames):
        path = os.path.join(save_dir, fname)
        if not os.path.exists(path + ".npz"):
            return None  # File thiếu → train lại toàn bộ

        # Nạp meta
        meta = np.load(path + "_meta.npz", allow_pickle=True)
        mn = float(meta["mn"][0])
        mx = float(meta["mx"][0])

        # Tạo model đúng loại
        model = create_model(method)
        model.load(path)

        # Rebuild sequences từ records
        runs = node_runs.get(seg, [])
        speeds = [v for run in runs for v in run]
        built = build_train_test(runs, SEQ_LEN, TRAIN_RATIO)
        if built is None:
            continue
        X_tr, y_tr, X_te, y_te = built
        X_tr_n, X_te_n, mn2, mx2 = normalize(X_tr, X_te)
        y_te_n = (y_te - mn) / (mx - mn + 1e-8)
        y_tr_n = (y_tr - mn) / (mx - mn + 1e-8)

        pred_n = model.predict(X_te_n)
        mae, rmse, mape, _, _ = compute_metrics(pred_n, y_te_n, mn, mx)
        road = next((r["road"] for r in records if r["segment"] == seg), "")

        result[seg] = {
            "model": model, "X_tr": X_tr_n, "y_tr": y_tr_n,
            "X_te": X_te_n, "y_te": y_te_n,
            "mn": mn, "mx": mx,
            "local_mae": mae, "local_rmse": rmse, "local_mape": mape,
            "road": road, "num_timesteps": len(speeds),
            "train_size": len(X_tr), "test_size": len(X_te),
        }

    if not result:
        return None
    print(f"        [LOAD] Nạp {len(result)} model từ {save_dir}/")
    return result


# =============================================================================
# 2. TRAIN CỤC BỘ TẤT CẢ NODE
# =============================================================================

def train_local_all(
    method: str,
    records: list,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Train model cục bộ tại từng node (segment) cho một phương pháp.

    Logic (cho GRU / LSTM / BiLSTM):
      1. Lấy chuỗi tốc độ của từng segment
      2. Tạo sequences (X, y) sliding window SEQ_LEN bước, TRONG TỪNG NGÀY
         (không vắt qua ranh giới ngày)
      3. Split theo ngày: 80% ngày đầu train / 20% ngày cuối test
      4. Normalize Min-Max trên tập train (tránh data leakage)
      5. Tạo model → train LOCAL_EPOCHS epoch
      6. Đánh giá MAE, RMSE, MAPE trên tập test

    Trả về dict { segment → {model, X_tr, y_tr, X_te, y_te, mn, mx, ...} }
    """
    node_runs = get_node_day_series(records)
    result: Dict[str, dict] = {}

    for seg, runs in sorted(node_runs.items()):
        speeds = [v for run in runs for v in run]
        built = build_train_test(runs, SEQ_LEN, TRAIN_RATIO)
        if built is None:
            continue
        X_tr, y_tr, X_te, y_te = built

        X_tr_n, X_te_n, mn, mx = normalize(X_tr, X_te)
        y_tr_n = (y_tr - mn) / (mx - mn + 1e-8)
        y_te_n = (y_te - mn) / (mx - mn + 1e-8)

        model = create_model(method)
        _train_model(model, X_tr_n, y_tr_n, epochs=_cfg.LOCAL_EPOCHS)
        pred_n = model.predict(X_te_n)
        mae, rmse, mape, _, _ = compute_metrics(pred_n, y_te_n, mn, mx)

        if verbose:
            print(f"        [{method.upper():8s}] [{seg[:35]:35s}] "
                  f"MAE={mae:5.2f}  RMSE={rmse:5.2f}  "
                  f"MAPE={mape:5.1f}%  T={len(speeds)}")

        road = next((r["road"] for r in records if r["segment"] == seg), "")
        result[seg] = {
            "model": model, "X_tr": X_tr_n, "y_tr": y_tr_n,
            "X_te": X_te_n, "y_te": y_te_n,
            "mn": mn, "mx": mx,
            "local_mae": mae, "local_rmse": rmse, "local_mape": mape,
            "road": road, "num_timesteps": len(speeds),
            "train_size": len(X_tr), "test_size": len(X_te),
        }

    return result


# =============================================================================
# 3. PIPELINE ĐẦY ĐỦ CHO 1 PHƯƠNG PHÁP × 1 SESSION
# =============================================================================

def run_one_method(
    method:      str,
    session_key: str,
    records:     list,
    verbose:     bool = True,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline đầy đủ cho 1 phương pháp × 1 khung giờ:

    Cache: nếu saved_models/<session>/<method>/ có đủ model + _run_info.json
           (kết quả local/FL lúc train) và khớp tập node hiện tại
           → nạp thẳng, BỎ QUA cả train local lẫn FL, trả lại đúng số liệu cũ.
    Không có cache (hoặc force_retrain=True):
      Bước 1: Train cục bộ tại từng node → local MAE/RMSE/MAPE
      Bước 2: Federated Learning → global model → fl MAE/RMSE/MAPE
      Bước 3: Lưu model + _run_info.json vào saved_models/
    """
    label = METHOD_NAMES.get(method, method.upper())
    print(f"\n    ── {label} ──")

    # ── Thử nạp cache ──
    node_data = None
    cached    = None
    if not force_retrain:
        cached = load_run_info(session_key, method)
        if cached:
            node_data = load_node_models(session_key, method, records)
            fl_segs = {m["segment"] for m in cached.get("fl_metrics", [])}
            if node_data is None or fl_segs != set(node_data):
                node_data, cached = None, None   # lệch dữ liệu → train lại

    if cached:
        print("    [CACHED] Nạp model + kết quả đã lưu (bỏ qua train local và FL)")
        local_metric_rows = cached["local_metrics"]
        fl_metrics        = cached["fl_metrics"]
        round_log         = cached["round_log"]
        fl_rounds_used    = cached.get("fl_rounds", len(round_log))
    else:
        # ── Bước 1: Local training ──
        print(f"    [1/2] Train cục bộ ({_cfg.LOCAL_EPOCHS} epochs)...")
        node_data = train_local_all(method, records, verbose)
        if not node_data:
            print("    [LỖI] Không có node nào đủ dữ liệu.")
            return {}

        # ── Bước 2: Federated Learning ──
        print(f"    [2/2] Federated Learning ({_cfg.FL_ROUNDS} vòng, "
              f"{_cfg.FL_LOCAL_EPOCHS} epochs/vòng)...")
        fl_metrics, round_log = run_federated(
            method=method, node_data=node_data,
            fl_rounds=_cfg.FL_ROUNDS, local_epochs=_cfg.FL_LOCAL_EPOCHS,
            verbose=verbose)
        if not fl_metrics:
            return {}

        info = build_run_info(method, node_data, fl_metrics, round_log)
        local_metric_rows = info["local_metrics"]
        fl_rounds_used    = info["fl_rounds"]

        # ── Lưu model sau FL + kết quả ──
        save_node_models(node_data, session_key, method, run_info=info)

    avg_mae_local  = round(float(np.mean([r["MAE_local"]  for r in local_metric_rows])), 3)
    avg_rmse_local = round(float(np.mean([r["RMSE_local"] for r in local_metric_rows])), 3)
    avg_mape_local = round(float(np.mean([r["MAPE_local"] for r in local_metric_rows])), 3)
    avg_mae_fl  = round(float(np.mean([m["MAE_fl"]  for m in fl_metrics])), 3)
    avg_rmse_fl = round(float(np.mean([m["RMSE_fl"] for m in fl_metrics])), 3)
    avg_mape_fl = round(float(np.mean([m["MAPE_fl"] for m in fl_metrics])), 3)
    improve     = round(avg_mae_local - avg_mae_fl, 3)

    print(f"    → Local: MAE={avg_mae_local:.2f} km/h  "
          f"RMSE={avg_rmse_local:.2f}  MAPE={avg_mape_local:.1f}%")
    print(f"    → FL   : MAE={avg_mae_fl:.2f} km/h  "
          f"RMSE={avg_rmse_fl:.2f}  MAPE={avg_mape_fl:.1f}%  "
          f"Δ MAE={improve:+.2f}")

    return {
        "method": method, "session": session_key,
        "num_nodes": len(fl_metrics), "fl_rounds": fl_rounds_used,
        "avg_MAE_local":  avg_mae_local,  "avg_RMSE_local": avg_rmse_local,
        "avg_MAPE_local": avg_mape_local,
        "avg_MAE_fl":     avg_mae_fl,     "avg_RMSE_fl":    avg_rmse_fl,
        "avg_MAPE_fl":    avg_mape_fl,
        "improvement_MAE": improve,
        "local_metrics": local_metric_rows,
        "fl_metrics":    fl_metrics,
        "round_log":     round_log,
    }
