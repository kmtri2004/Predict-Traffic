"""
=============================================================================
forecaster.py — Dự báo tốc độ theo yêu cầu (BiLSTM + SCAFFOLD)
=============================================================================
Cách dùng:
    forecaster.load_all()     # gọi 1 LẦN lúc server khởi động (nạp model + dữ liệu)
    forecaster.meta()         # danh sách buổi / ngày / đoạn đường
    forecaster.forecast(...)  # dự báo cho 1 buổi × 1 ngày × 1 đoạn đường

Ý tưởng:
  - Dữ liệu có bước 5 phút. Model học dự báo 1 bước (5 phút) từ SEQ_LEN điểm trước.
  - "Đứng ở" giờ gốc t (t đủ SEQ_LEN điểm lịch sử, ví dụ 6:40 khi buổi bắt đầu
    6:30) → dự báo t+5, t+10, ... bằng cách đệ quy: ghép dự báo vừa ra vào cửa
    sổ rồi dự báo tiếp.
  - Với mỗi giờ gốc, nếu t+h nằm trong dữ liệu thật → trả thêm "actual" để so sánh.
=============================================================================
"""

import threading
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import SEQ_LEN, STEP_MINUTES, TRAIN_RATIO, SESSION_LABEL
from data_loader import load_all_sessions, load_snapshots, extract_records
from train_evaluate import load_node_models

MODEL_METHOD = "bilstm"
DEFAULT_HORIZONS = (5, 10, 15, 20, 25, 30)
MAX_HORIZON = 60

_lock = threading.Lock()          # bảo vệ model khi nhiều request cùng lúc
STATE: Dict[str, dict] = {}       # STATE[session] = {nodes, runs, freeflow, test_dates, train_dates}


# =============================================================================
# TIỆN ÍCH
# =============================================================================

def classify_congestion(speed: float, free_flow: float) -> str:
    """Giống predict.py: ratio = tốc độ / freeFlow."""
    if free_flow <= 0:
        return "KHÔNG RÕ"
    ratio = speed / free_flow
    if ratio >= 0.80:
        return "THÔNG THOÁNG"
    if ratio >= 0.60:
        return "CHẬM"
    if ratio >= 0.40:
        return "KẸT"
    return "KẸT NẶNG"


def _add_min(hhmm: str, minutes: int) -> str:
    h, m = map(int, hhmm.split(":"))
    t = h * 60 + m + int(minutes)
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


def _day_runs(records: List[dict], max_gap_min: float = 7.5) -> Dict[str, List[dict]]:
    """
    Giống get_node_day_series (data_loader) nhưng GIỮ LẠI ngày + giờ.
    Một run mới bắt đầu khi sang ngày khác hoặc thiếu snapshot (> max_gap_min).
    Cách tách này phải y hệt lúc train để chia train/test khớp.
    """
    runs: Dict[str, List[dict]] = defaultdict(list)
    last: Dict[str, datetime] = {}
    for r in records:                                   # records đã sắp theo thời gian
        seg = r["segment"]
        ts = datetime.fromisoformat(r["timestamp_dt"])
        prev = last.get(seg)
        if (prev is None or ts.date() != prev.date()
                or (ts - prev).total_seconds() / 60.0 > max_gap_min):
            runs[seg].append({"date": ts.strftime("%Y%m%d"), "times": [], "speeds": []})
        runs[seg][-1]["times"].append(ts.strftime("%H:%M"))
        runs[seg][-1]["speeds"].append(float(r["currentSpeed"]))
        last[seg] = ts
    return runs


def _freeflow(records: List[dict]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        ff = r.get("freeFlowSpeed", 0)
        if ff > 0:
            acc[r["segment"]].append(ff)
    return {seg: float(np.mean(v)) for seg, v in acc.items()}


# =============================================================================
# NẠP 1 LẦN LÚC KHỞI ĐỘNG
# =============================================================================

def load_all(verbose: bool = True) -> None:
    STATE.clear()
    _forecast_cached.cache_clear()

    for sk, files in load_all_sessions().items():
        if not files:
            continue
        records = extract_records(load_snapshots(files))
        nodes = load_node_models(sk, MODEL_METHOD, records)
        if not nodes:
            print(f"[forecaster] {sk}: không nạp được model "
                  f"(thiếu saved_models hoặc dữ liệu không khớp)")
            continue

        runs = _day_runs(records)
        by_seg: Dict[str, Dict[str, tuple]] = {}
        test_dates, all_dates = set(), set()

        for seg, rs in runs.items():
            if seg not in nodes:
                continue
            rs = [r for r in rs if len(r["speeds"]) > SEQ_LEN]     # như build_train_test
            if len(rs) < 2:
                continue
            k = min(max(1, int(len(rs) * TRAIN_RATIO)), len(rs) - 1)  # cùng cách chia ngày
            by_seg[seg] = {}
            for i, r in enumerate(rs):
                old = by_seg[seg].get(r["date"])
                if old is None or len(r["speeds"]) > len(old[0]["speeds"]):
                    by_seg[seg][r["date"]] = (r, i >= k)
                all_dates.add(r["date"])
                if i >= k:
                    test_dates.add(r["date"])

        STATE[sk] = {
            "nodes": nodes,
            "runs": by_seg,
            "freeflow": _freeflow(records),
            "test_dates": sorted(test_dates),
            "train_dates": sorted(all_dates - test_dates),
        }
        if verbose:
            print(f"[forecaster] {sk}: {len(by_seg)} đoạn đường, "
                  f"{len(test_dates)} ngày test, {len(all_dates - test_dates)} ngày train")


def meta() -> dict:
    out = {}
    for sk, s in STATE.items():
        out[sk] = {
            "label": SESSION_LABEL.get(sk, sk),
            "test_dates": s["test_dates"],
            "train_dates": s["train_dates"],
            "segments": sorted(
                ({"segment": seg, "road": s["nodes"][seg]["road"]} for seg in s["runs"]),
                key=lambda x: (x["road"], x["segment"])),
        }
    return {
        "model": "BiLSTM + SCAFFOLD",
        "seq_len": SEQ_LEN,
        "step_minutes": STEP_MINUTES,
        "default_horizons": list(DEFAULT_HORIZONS),
        "sessions": out,
    }


# =============================================================================
# DỰ BÁO
# =============================================================================

def forecast(session: str, date: str, segment: str,
             horizons: Sequence[int] = DEFAULT_HORIZONS,
             origin: Optional[str] = None) -> dict:
    """
    session : "sang" | "trua" | "chieu"
    date    : "YYYYMMDD"
    segment : tên đoạn đường (đúng như trong /meta)
    horizons: các mốc phút, bội của 5, 5..60
    origin  : "HH:MM" để chỉ lấy 1 giờ gốc (ví dụ "06:40"); None = mọi giờ gốc
    Ném KeyError nếu không có dữ liệu, ValueError nếu tham số sai.
    """
    hs = sorted({int(h) for h in horizons})
    if not hs or any(h < STEP_MINUTES or h > MAX_HORIZON or h % STEP_MINUTES for h in hs):
        raise ValueError(f"horizon phải là bội của {STEP_MINUTES}, trong "
                         f"{STEP_MINUTES}–{MAX_HORIZON} phút")
    return _forecast_cached(session, date, segment, tuple(hs), origin)


@lru_cache(maxsize=1024)
def _forecast_cached(session: str, date: str, segment: str,
                     hs: tuple, origin: Optional[str]) -> dict:
    if session not in STATE:
        raise KeyError(f"Không có buổi '{session}'")
    s = STATE[session]
    if segment not in s["runs"]:
        raise KeyError(f"Không có đoạn đường '{segment}'")
    if date not in s["runs"][segment]:
        raise KeyError(f"Đoạn đường này không có dữ liệu ngày {date}")

    d = s["nodes"][segment]
    run, is_test = s["runs"][segment][date]
    sp = np.array(run["speeds"], dtype=np.float32)
    times = run["times"]
    mn, mx = d["mn"], d["mx"]
    rng = (mx - mn) if mx != mn else 1.0
    ff = s["freeflow"].get(segment, 0.0)

    steps = [h // STEP_MINUTES for h in hs]
    max_step = max(steps)

    origins = list(range(SEQ_LEN - 1, len(sp)))       # giờ gốc sớm nhất = điểm thứ SEQ_LEN
    if origin is not None:
        origins = [i for i in origins if times[i] == origin]
        if not origins:
            first = times[SEQ_LEN - 1]
            raise KeyError(f"Giờ gốc {origin} không hợp lệ "
                           f"(sớm nhất {first}, cần đủ {SEQ_LEN} điểm lịch sử)")

    # Dự báo đệ quy, tính cùng lúc cho mọi giờ gốc (batch)
    win = np.stack([(sp[i - SEQ_LEN + 1:i + 1] - mn) / rng
                    for i in origins]).astype(np.float32)
    path = []
    with _lock:
        for _ in range(max_step):
            p = d["model"].predict(win)
            p = np.clip(np.asarray(p, dtype=np.float32), -0.1, 1.1)   # chống trôi khi đệ quy
            path.append(p)
            win = np.concatenate([win[:, 1:], p[:, None]], axis=1)
    pred_kmh = np.maximum(np.stack(path, axis=1) * rng + mn, 0.0)     # [n_origin, max_step]

    points = []
    for j, i in enumerate(origins):
        fc = {}
        for h, st in zip(hs, steps):
            actual = float(sp[i + st]) if i + st < len(sp) else None
            pred = round(float(pred_kmh[j, st - 1]), 1)
            fc[str(h)] = {
                "time": _add_min(times[i], h),
                "predicted": pred,
                "status": classify_congestion(pred, ff),
                "actual": None if actual is None else round(actual, 1),
            }
        cur = round(float(sp[i]), 1)
        points.append({
            "origin_time": times[i],
            "current": cur,
            "current_status": classify_congestion(cur, ff),
            "forecast": fc,
        })

    mae, n_eval = {}, {}
    for h in hs:
        e = [abs(p["forecast"][str(h)]["predicted"] - p["forecast"][str(h)]["actual"])
             for p in points if p["forecast"][str(h)]["actual"] is not None]
        mae[str(h)] = round(float(np.mean(e)), 2) if e else None
        n_eval[str(h)] = len(e)

    return {
        "session": session,
        "date": date,
        "segment": segment,
        "road": d["road"],
        "is_test_day": bool(is_test),
        "freeflow_speed": round(ff, 1),
        "step_minutes": STEP_MINUTES,
        "seq_len": SEQ_LEN,
        "first_origin": times[SEQ_LEN - 1],
        "horizons": list(hs),
        "points": points,
        "mae_by_horizon": mae,
        "n_eval_by_horizon": n_eval,
        "note": ("Model chỉ học dự báo 1 bước (5 phút); các mốc xa hơn là dự báo "
                 "đệ quy nên sai số tăng dần theo horizon."),
    }
