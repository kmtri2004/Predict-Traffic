"""
=============================================================================
models.py  —  Định nghĩa các mô hình dự báo lưu lượng giao thông
=============================================================================
Đồ án: Mô phỏng kiến trúc Node-Agent sử dụng Học Liên Bang

VAI TRÒ: Định nghĩa kiến trúc 3 mô hình học máy:
  1. GRUModel     — Gated Recurrent Unit (đơn hướng)
  2. LSTMModel    — Long Short-Term Memory (đơn hướng)
  3. BiLSTMModel  — Bidirectional LSTM (★ mô hình tốt nhất)

  Tất cả mô hình đều có interface thống nhất:
    predict(X)      → np.ndarray   (dự báo)
    get_weights()   → dict         (lấy tham số — dùng cho FL)
    set_weights(d)  → None         (nạp tham số — dùng cho FL)
    save(path)      → None         (lưu tham số ra file .npz)
    load(path)      → None         (nạp tham số từ file .npz)

  Interface thống nhất giúp Federated Learning (federated.py) có thể
  làm việc với bất kỳ mô hình nào mà không cần biết chi tiết bên trong.
=============================================================================
"""

import os
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from config import (
    HIDDEN_SIZE, NUM_LAYERS, LEARNING_RATE, LOCAL_EPOCHS,
    RANDOM_SEED,
)

# ── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    torch.manual_seed(RANDOM_SEED)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[CẢNH BÁO] PyTorch chưa cài → pip install torch")

np.random.seed(RANDOM_SEED)


# =============================================================================
# HELPER
# =============================================================================

def _to_tensor(X: np.ndarray) -> "torch.Tensor":
    """[N, seq_len] → [N, seq_len, 1]  (thêm chiều feature)."""
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1)


def _to_y_tensor(y: np.ndarray) -> "torch.Tensor":
    """[N] → [N, 1]."""
    return torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


# =============================================================================
# MÔ HÌNH 1: GRU
# =============================================================================

class GRUModel(nn.Module if HAS_TORCH else object):
    """
    Gated Recurrent Unit — dự báo tốc độ theo chuỗi thời gian.

    Kiến trúc:
        Input  : [batch, seq_len, 1]   — chuỗi tốc độ đã chuẩn hoá
        GRU    : hidden_size=32, 1 layer
        Linear : 32 → 1
        Output : [batch, 1]            — tốc độ bước tiếp theo

    So với LSTM: ít tham số hơn, train nhanh hơn, phù hợp tập dữ liệu
    vừa (~30 ngày). Không có cell state riêng nên ít khả năng nhớ
    dài hạn hơn BiLSTM.
    """

    def __init__(self, input_size: int = 1,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS):
        if HAS_TORCH:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self._numpy_init(hidden_size)

    def _numpy_init(self, hs: int):
        rng = np.random.default_rng(RANDOM_SEED)
        s = 0.1
        self.Wr = rng.normal(0, s, (hs, 1));  self.Ur = rng.normal(0, s, (hs, hs)); self.br = np.zeros(hs)
        self.Wz = rng.normal(0, s, (hs, 1));  self.Uz = rng.normal(0, s, (hs, hs)); self.bz = np.zeros(hs)
        self.Wn = rng.normal(0, s, (hs, 1));  self.Un = rng.normal(0, s, (hs, hs)); self.bn = np.zeros(hs)
        self.Wo = rng.normal(0, s, (1, hs));  self.bo = np.zeros(1)
        self._hs = hs

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def _forward_numpy(self, seq: np.ndarray) -> float:
        sig  = lambda v: 1 / (1 + np.exp(-np.clip(v, -20, 20)))
        tanh = lambda v: np.tanh(np.clip(v, -20, 20))
        h = np.zeros(self._hs)
        for xt in seq:
            x = np.array([float(xt)])
            r = sig(self.Wr @ x + self.Ur @ h + self.br)
            z = sig(self.Wz @ x + self.Uz @ h + self.bz)
            n = tanh(self.Wn @ x + self.Un @ (r * h) + self.bn)
            h = (1 - z) * n + z * h
        return float((self.Wo @ h + self.bo).flat[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if HAS_TORCH:
            self.eval()
            with torch.no_grad():
                return self.forward(_to_tensor(X)).squeeze(-1).numpy()
        return np.array([self._forward_numpy(row) for row in X])

    def get_weights(self) -> dict:
        if HAS_TORCH:
            return {k: v.cpu().numpy().copy() for k, v in self.state_dict().items()}
        return {k: getattr(self, k).copy()
                for k in ["Wr","Ur","br","Wz","Uz","bz","Wn","Un","bn","Wo","bo"]}

    def set_weights(self, d: dict):
        if HAS_TORCH:
            self.load_state_dict({k: torch.tensor(v) for k, v in d.items()})
        else:
            for k, v in d.items(): setattr(self, k, v.copy())

    def save(self, path: str) -> None:
        """Lưu weights ra file .npz để tái sử dụng không cần train lại."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.get_weights())

    def load(self, path: str) -> None:
        """Nạp weights từ file .npz đã lưu."""
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        self.set_weights({k: data[k] for k in data.files})


# =============================================================================
# MÔ HÌNH 2: LSTM
# =============================================================================

class LSTMModel(nn.Module if HAS_TORCH else object):
    """
    Long Short-Term Memory — mở rộng GRU với Cell State.

    Kiến trúc:
        Input  : [batch, seq_len, 1]
        LSTM   : hidden_size=32, 1 layer
        Linear : 32 → 1
        Output : [batch, 1]

    LSTM có thêm Cell State (c_t) song song với Hidden State (h_t),
    cho phép ghi nhớ/quên thông tin tùy ý qua các cổng (gate).
    Phức tạp hơn GRU nhưng thường tốt hơn trên chuỗi dài hơn.
    """

    def __init__(self, input_size: int = 1,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS):
        if HAS_TORCH:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self._hs = hidden_size
            rng = np.random.default_rng(RANDOM_SEED)
            s = 0.1
            self.Wi = rng.normal(0, s, (hidden_size, 1+hidden_size)); self.bi = np.zeros(hidden_size)
            self.Wf = rng.normal(0, s, (hidden_size, 1+hidden_size)); self.bf = np.zeros(hidden_size)
            self.Wo = rng.normal(0, s, (hidden_size, 1+hidden_size)); self.bo = np.zeros(hidden_size)
            self.Wg = rng.normal(0, s, (hidden_size, 1+hidden_size)); self.bg = np.zeros(hidden_size)
            self.Wout = rng.normal(0, s, (1, hidden_size));            self.bout = np.zeros(1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def _forward_numpy(self, seq: np.ndarray) -> float:
        sig  = lambda v: 1 / (1 + np.exp(-np.clip(v, -20, 20)))
        tanh = lambda v: np.tanh(np.clip(v, -20, 20))
        h = np.zeros(self._hs); c = np.zeros(self._hs)
        for xt in seq:
            z = np.concatenate([[float(xt)], h])
            i_gate = sig(self.Wi @ z + self.bi)
            f_gate = sig(self.Wf @ z + self.bf)
            o_gate = sig(self.Wo @ z + self.bo)
            g_gate = tanh(self.Wg @ z + self.bg)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * tanh(c)
        return float((self.Wout @ h + self.bout).flat[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if HAS_TORCH:
            self.eval()
            with torch.no_grad():
                return self.forward(_to_tensor(X)).squeeze(-1).numpy()
        return np.array([self._forward_numpy(row) for row in X])

    def get_weights(self) -> dict:
        if HAS_TORCH:
            return {k: v.cpu().numpy().copy() for k, v in self.state_dict().items()}
        return {k: getattr(self, k).copy()
                for k in ["Wi","bi","Wf","bf","Wo","bo","Wg","bg","Wout","bout"]}

    def set_weights(self, d: dict):
        if HAS_TORCH:
            self.load_state_dict({k: torch.tensor(v) for k, v in d.items()})
        else:
            for k, v in d.items(): setattr(self, k, v.copy())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.get_weights())

    def load(self, path: str) -> None:
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        self.set_weights({k: data[k] for k in data.files})


# =============================================================================
# MÔ HÌNH 3: BiLSTM  ★ MÔ HÌNH TỐT NHẤT TRONG ĐỒ ÁN
# =============================================================================

class BiLSTMModel(nn.Module if HAS_TORCH else object):
    """
    Bidirectional LSTM — LSTM 2 chiều (forward + backward).

    Kiến trúc:
        Input      : [batch, seq_len, 1]
        BiLSTM     : hidden_size=32, bidirectional=True  → output 64 chiều
        Linear     : 64 → 1
        Output     : [batch, 1]

    Ưu điểm so với LSTM đơn hướng:
      - Nhìn chuỗi theo cả 2 chiều: trái → phải (forward) và
        phải → trái (backward).
      - Với chuỗi thời gian giao thông, chiều backward học được
        "pattern kết thúc" của chuỗi — tốc độ đang tăng hay giảm.
      - Kết hợp 2 chiều → biểu diễn phong phú hơn → MAE thấp hơn.

    Kết quả thực nghiệm (SEQ_LEN=8, 30 ngày dữ liệu):
      MAE sau FL: Sáng 3.02 | Trưa 2.49 | Chiều 1.73 km/h
      MAPE:       Sáng 10.2%| Trưa 9.0% | Chiều 7.6%

    Lý do tốt hơn GRU/LSTM trong môi trường FL:
      Biểu diễn 2 chiều tổng quát hơn → ít bị "client drift"
      khi FedAvg trộn weights từ 23 node có pattern khác nhau (Non-IID).
    """

    def __init__(self, input_size: int = 1,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS):
        if HAS_TORCH:
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=True,
            )
            self.fc = nn.Linear(hidden_size * 2, 1)  # *2 vì bidirectional
        else:
            # Fallback: 2 LSTM đơn hướng
            self._fwd = LSTMModel(input_size, hidden_size, num_layers)
            self._bwd = LSTMModel(input_size, hidden_size, num_layers)
            self._hs = hidden_size
            rng = np.random.default_rng(RANDOM_SEED)
            self.Wout = rng.normal(0, 0.1, (1, hidden_size * 2))
            self.bout  = np.zeros(1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1, :])   # lấy hidden state bước cuối

    def _forward_numpy(self, seq: np.ndarray) -> float:
        fwd_h = self._fwd._forward_numpy(seq)
        bwd_h = self._bwd._forward_numpy(seq[::-1])
        h = np.array([fwd_h, bwd_h])
        return float((self.Wout @ np.concatenate([
            np.array([fwd_h] * self._hs),
            np.array([bwd_h] * self._hs)
        ]) + self.bout).flat[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if HAS_TORCH:
            self.eval()
            with torch.no_grad():
                return self.forward(_to_tensor(X)).squeeze(-1).numpy()
        return np.array([self._forward_numpy(row) for row in X])

    def get_weights(self) -> dict:
        if HAS_TORCH:
            return {k: v.cpu().numpy().copy() for k, v in self.state_dict().items()}
        d = {}
        for k, v in self._fwd.get_weights().items(): d[f"fwd_{k}"] = v
        for k, v in self._bwd.get_weights().items(): d[f"bwd_{k}"] = v
        d["Wout"] = self.Wout.copy(); d["bout"] = self.bout.copy()
        return d

    def set_weights(self, d: dict):
        if HAS_TORCH:
            self.load_state_dict({k: torch.tensor(v) for k, v in d.items()})
        else:
            fwd_d = {k[4:]: v for k, v in d.items() if k.startswith("fwd_")}
            bwd_d = {k[4:]: v for k, v in d.items() if k.startswith("bwd_")}
            self._fwd.set_weights(fwd_d)
            self._bwd.set_weights(bwd_d)
            if "Wout" in d: self.Wout = d["Wout"].copy()
            if "bout" in d: self.bout = d["bout"].copy()

    def save(self, path: str) -> None:
        """
        Lưu weights ra file .npz.

        Sau khi train + FL xong, gọi save() để không phải train lại lần sau.
        Dùng load() để nạp lại và dự báo ngay.

        Ví dụ:
            model.save("saved_models/sang/bilstm/TruongChinh")
            # → tạo file saved_models/sang/bilstm/TruongChinh.npz
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.get_weights())

    def load(self, path: str) -> None:
        """
        Nạp weights từ file .npz.

        Ví dụ:
            model = BiLSTMModel()
            model.load("saved_models/sang/bilstm/TruongChinh")
            pred = model.predict(X_input)
        """
        fpath = path + ".npz" if not path.endswith(".npz") else path
        data = np.load(fpath)
        self.set_weights({k: data[k] for k in data.files})


# =============================================================================
# FACTORY
# =============================================================================

def create_model(method: str, **kwargs):
    """
    Tạo model theo tên phương pháp.

    Tham số:
        method : "gru" | "lstm" | "bilstm"

    Ví dụ:
        model = create_model("bilstm")
    """
    method = method.lower()
    if method == "gru":
        return GRUModel()
    elif method == "lstm":
        return LSTMModel()
    elif method == "bilstm":
        return BiLSTMModel()
    else:
        raise ValueError(f"Phương pháp không hỗ trợ: {method}. "
                         f"Chọn: gru | lstm | bilstm")


METHOD_NAMES = {
    "gru":    "GRU",
    "lstm":   "LSTM",
    "bilstm": "BiLSTM",
}