"""
=============================================================================
data_loader.py  —  Đọc & xử lý dữ liệu JSON TomTom
=============================================================================
Chức năng:
  1. load_all_sessions()   — tìm & nhóm file JSON theo S/T/C
  2. load_snapshots()      — đọc danh sách file JSON → list dict
  3. extract_records()     — flatten JSON → list bản ghi phẳng
  4. get_node_series()     — dict {segment → [speed_t0, ...]}
  5. make_sequences()      — tạo cặp (X, y) dạng sliding window
  6. normalize() / denorm()— chuẩn hoá min-max
  7. write_csv()           — lưu CSV
=============================================================================
"""

import csv
import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    DATA_DIR, SESSION_MAP, SESSION_LABEL, SEQ_LEN,
    TRAIN_RATIO, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. TÌM FILE JSON
# =============================================================================

def load_all_sessions() -> Dict[str, List[str]]:
    """
    Quét DATA_DIR để tìm tất cả file JSON theo quy tắc đặt tên.

    Trả về dict:
        {
          "sang":  ["data_reduce_20260406_S.json", ...],
          "trua":  [...],
          "chieu": [...],
        }
    """
    result: Dict[str, List[str]] = {"sang": [], "trua": [], "chieu": []}
    pattern = os.path.join(DATA_DIR, "data_reduce_*.json")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        # Lấy ký tự cuối trước ".json": S, T hoặc C
        suffix = basename.replace(".json", "").split("_")[-1].upper()
        if suffix in SESSION_MAP:
            result[SESSION_MAP[suffix]].append(path)
    return result


# =============================================================================
# 2. ĐỌC SNAPSHOTS
# =============================================================================

def load_snapshots(paths: List[str]) -> List[dict]:
    """Đọc nhiều file JSON → list snapshot (mỗi snapshot = 1 dict)."""
    snaps = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            snaps.extend(raw)
        else:
            snaps.append(raw)
    return snaps


# =============================================================================
# 3. CHUYỂN JSON → BẢN GHI PHẲNG
# =============================================================================

def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Khoảng cách Haversine giữa 2 toạ độ (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def extract_records(snapshots: List[dict]) -> List[dict]:
    """
    Flatten JSON TomTom → list bản ghi phẳng.

    Mỗi bản ghi đại diện cho 1 segment tại 1 thời điểm:
        timestamp, road, segment, currentSpeed (km/h),
        freeFlowSpeed, congestion_ratio, delay_sec, length_km, …

    Dữ liệu được sắp xếp theo thứ tự thời gian để đảm bảo
    chuỗi thời gian (time series) đúng thứ tự khi tạo sequences.
    """
    records = []
    for snap in snapshots:
        ts_str = snap["snapshot_timestamp"]
        try:
            ts = datetime.strptime(ts_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            ts = datetime.strptime(ts_str, "%d/%m/%Y %H:%M")

        for road, rd in snap["flow_data"].items():
            direction = rd.get("direction", "")
            for seg, sd in rd["flow_data"].items():
                fsd    = sd["flowSegmentData"]
                coords = fsd["coordinates"]["coordinate"]

                cur_speed  = float(fsd["currentSpeed"])
                free_speed = float(fsd["freeFlowSpeed"])
                congestion = cur_speed / free_speed if free_speed > 0 else 0.0

                length_km = _haversine_km(
                    coords[0]["latitude"],  coords[0]["longitude"],
                    coords[-1]["latitude"], coords[-1]["longitude"],
                )

                records.append({
                    "timestamp":         ts_str,
                    "timestamp_dt":      ts.isoformat(),
                    "date":              ts.strftime("%d/%m/%Y"),
                    "hour":              ts.hour,
                    "minute":            ts.minute,
                    "road":              road,
                    "segment":           seg,
                    "direction":         direction,
                    "frc":               fsd["frc"],
                    "currentSpeed":      cur_speed,
                    "freeFlowSpeed":     free_speed,
                    "currentTravelTime": fsd["currentTravelTime"],
                    "freeFlowTravelTime":fsd["freeFlowTravelTime"],
                    "congestion_ratio":  round(congestion, 4),
                    "delay_sec":         fsd["currentTravelTime"] - fsd["freeFlowTravelTime"],
                    "roadClosure":       int(fsd["roadClosure"]),
                    "length_km":         round(length_km, 4),
                    "start_lat":         coords[0]["latitude"],
                    "start_lon":         coords[0]["longitude"],
                    "end_lat":           coords[-1]["latitude"],
                    "end_lon":           coords[-1]["longitude"],
                })

    records.sort(key=lambda r: (r["timestamp_dt"], r["road"], r["segment"]))
    return records


# =============================================================================
# 5. CHUỖI THỜI GIAN
# =============================================================================

def get_node_series(records: List[dict]) -> Dict[str, List[float]]:
    """
    Trích xuất chuỗi tốc độ theo thời gian cho từng segment.

    Trả về:
        { "Trường Chinh giao Cộng Hòa": [22.0, 21.5, 23.0, ...], ... }

    Đây là dữ liệu đầu vào cho GRU/LSTM/BiLSTM và Q-Learning.
    """
    series: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        series[r["segment"]].append(float(r["currentSpeed"]))
    return dict(series)


def make_sequences(speeds: List[float],
                   seq_len: int = SEQ_LEN
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo cặp (X, y) bằng sliding window:

        X[i] = [v_t,   v_{t+1}, ..., v_{t+seq_len-1}]  ← 6 bước lịch sử
        y[i] = v_{t+seq_len}                             ← 1 bước tương lai

    Ví dụ với seq_len=6, speeds=[22,21,20,23,24,25,22,21]:
        X[0]=[22,21,20,23,24,25]  →  y[0]=22
        X[1]=[21,20,23,24,25,22]  →  y[1]=21

    Đây là dạng "one-step-ahead forecast" — dự báo tốc độ
    ngay bước tiếp theo dựa trên 6 bước gần nhất.
    """
    X, y = [], []
    for i in range(len(speeds) - seq_len):
        X.append(speeds[i: i + seq_len])
        y.append(speeds[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_node_day_series(records: List[dict],
                        max_gap_min: float = 7.5) -> Dict[str, List[List[float]]]:
    """
    Chuỗi tốc độ của từng segment, TÁCH THEO NGÀY.

    Trả về: { segment: [ [v_ngày1...], [v_ngày2...], ... ] }
    Một "run" mới bắt đầu khi sang ngày khác HOẶC khi thiếu snapshot
    (cách nhau > max_gap_min phút) -> cửa sổ trượt không bao giờ vắt qua
    ranh giới ngày / lỗ hổng dữ liệu.
    """
    runs: Dict[str, List[List[float]]] = defaultdict(list)
    last_ts: Dict[str, datetime] = {}
    for r in records:                       # records đã sắp theo thời gian
        seg = r["segment"]
        ts = datetime.fromisoformat(r["timestamp_dt"])
        prev = last_ts.get(seg)
        if (prev is None or ts.date() != prev.date()
                or (ts - prev).total_seconds() / 60.0 > max_gap_min):
            runs[seg].append([])
        runs[seg][-1].append(float(r["currentSpeed"]))
        last_ts[seg] = ts
    return dict(runs)


def build_train_test(runs: List[List[float]], seq_len: int = SEQ_LEN,
                     train_ratio: float = TRAIN_RATIO):
    """
    Tạo (X_tr, y_tr, X_te, y_te) từ các run (mỗi run = 1 ngày).
    - Cửa sổ chỉ nằm trong 1 run.
    - Chia train/test THEO NGÀY (các ngày đầu train, ngày cuối test),
      nên test không lẫn mẫu cùng ngày với train.
    Trả về None nếu không đủ dữ liệu.
    """
    runs = [r for r in runs if len(r) > seq_len]
    if len(runs) < 2:
        return None
    k = min(max(1, int(len(runs) * train_ratio)), len(runs) - 1)

    def _stack(rs):
        Xs, ys = [], []
        for r in rs:
            X, y = make_sequences(r, seq_len)
            Xs.append(X); ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    X_tr, y_tr = _stack(runs[:k])
    X_te, y_te = _stack(runs[k:])
    if len(X_tr) < 4 or len(X_te) < 1:
        return None
    return X_tr, y_tr, X_te, y_te


def normalize(train: np.ndarray,
              test: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
    """
    Chuẩn hoá Min-Max về [0, 1].

    Chỉ tính min/max trên tập train để tránh data leakage.
    Tập test dùng lại min/max của train.

    Trả về: (train_norm, test_norm, min_val, max_val)
    """
    mn, mx = float(train.min()), float(train.max())
    rng = mx - mn if mx != mn else 1.0
    train_n = (train - mn) / rng
    test_n  = (test  - mn) / rng if test is not None else None
    return train_n, test_n, mn, mx


def denormalize(arr_norm: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """Khôi phục giá trị thực từ dạng đã chuẩn hoá."""
    return arr_norm * (mx - mn) + mn


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                    mn: float, mx: float) -> Tuple[float, float, float,
                                                   np.ndarray, np.ndarray]:
    """
    Tính 3 chỉ số đánh giá trên giá trị thực (km/h):

        MAE  = mean |ŷ - y|
        RMSE = sqrt(mean (ŷ - y)²)
        MAPE = mean |ŷ - y| / |y| × 100  (%)  [bỏ qua y=0]

    Trả về: (MAE, RMSE, MAPE, y_pred_real, y_true_real)
    """
    y_pred_r = denormalize(y_pred, mn, mx)
    y_true_r = denormalize(y_true, mn, mx)
    diff  = y_pred_r - y_true_r
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    mask  = y_true_r != 0
    mape  = float(np.mean(np.abs(diff[mask] / y_true_r[mask])) * 100) if mask.any() else 0.0
    return round(mae, 3), round(rmse, 3), round(mape, 3), y_pred_r, y_true_r


# =============================================================================
# 6. GHI CSV
# =============================================================================

def write_csv(rows: List[dict], path: str) -> None:
    """Ghi list dict → CSV (tạo thư mục nếu chưa có)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# 7. TIỆN ÍCH KIỂM TRA
# =============================================================================

def summary_sessions(session_files: Dict[str, List[str]]) -> None:
    """In tóm tắt số file và ngày thu thập cho từng phiên."""
    print("\n  Dữ liệu TomTom tìm thấy:")
    for key in ["sang", "trua", "chieu"]:
        files = session_files.get(key, [])
        if not files:
            print(f"    {SESSION_LABEL[key]}: KHÔNG CÓ FILE")
            continue
        dates = sorted({os.path.basename(f).split("_")[2] for f in files})
        print(f"    {SESSION_LABEL[key]:35s}: "
              f"{len(files):2d} file  (ngày: {', '.join(dates)})")
