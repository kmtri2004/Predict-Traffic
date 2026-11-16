from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config as _cfg
from config import FL_ROUNDS, FL_LOCAL_EPOCHS, RANDOM_SEED
from data_loader import compute_metrics, make_sequences, normalize, write_csv

np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def fedavg_weights(weight_list: List[Dict[str, np.ndarray]],
                   data_sizes: List[int]) -> Dict[str, np.ndarray]:
    """Tổng hợp tại server: W_global[key] = Σ_i (n_i / N) · W_i[key]"""
    total = sum(data_sizes)
    if total == 0 or not weight_list:
        return deepcopy(weight_list[0]) if weight_list else {}

    global_weights: Dict[str, np.ndarray] = {}
    keys = list(weight_list[0].keys())

    for key in keys:
        stacked = np.stack([w[key].astype(np.float64) for w in weight_list], axis=0)
        coefs = np.array(data_sizes, dtype=np.float64) / total
        for _ in range(stacked.ndim - 1):
            coefs = coefs[:, np.newaxis]
        global_weights[key] = np.sum(stacked * coefs, axis=0).astype(np.float32)

    return global_weights


class FederatedNode:
    """1 node (1 segment) trong hệ thống FL."""

    def __init__(self, node_id: str, model: Any,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test:  np.ndarray, y_test:  np.ndarray,
                 mn: float, mx: float):
        self.node_id  = node_id
        self.model    = model
        self.X_train  = X_train
        self.y_train  = y_train
        self.X_test   = X_test
        self.y_test   = y_test
        self.mn       = mn
        self.mx       = mx
        self.n_train  = len(X_train)
        # Control variate cục bộ c_i (khởi tạo ở vòng đầu)
        self.c_i: Optional[Dict[str, np.ndarray]] = None
        self.c_i_next: Optional[Dict[str, np.ndarray]] = None

    def local_train(self, epochs: int = FL_LOCAL_EPOCHS,
                    lr: float = None,
                    correction: Optional[Dict[str, np.ndarray]] = None
                    ) -> None:
        """Fine-tune cục bộ; correction = c - c_i."""
        _train_model(self.model, self.X_train, self.y_train,
                     epochs=epochs, lr=lr, correction=correction)

    def full_gradient(self) -> Dict[str, np.ndarray]:
        """∇L_i(w) full-batch tại weights hiện tại của model."""
        return _full_gradient(self.model, self.X_train, self.y_train)

    def evaluate(self) -> Tuple[float, float, float]:
        pred_n = self.model.predict(self.X_test)
        mae, rmse, mape, _, _ = compute_metrics(
            pred_n, self.y_test, self.mn, self.mx)
        return mae, rmse, mape

    def get_weights(self) -> Dict[str, np.ndarray]:
        return self.model.get_weights()

    def set_weights(self, d: Dict[str, np.ndarray]) -> None:
        self.model.set_weights(d)


class FederatedServer:
    """Điều phối SCAFFOLD: gom weights → trung bình → phát → node train với gradient hiệu chỉnh."""

    def __init__(self, nodes: List[FederatedNode]):
        self.nodes      = nodes
        self.c: Optional[Dict[str, np.ndarray]] = None   # control variate server
        self.round_log  = []

    def aggregate(self) -> Dict[str, np.ndarray]:
        weight_list = [n.get_weights() for n in self.nodes]
        data_sizes  = [n.n_train for n in self.nodes]
        return fedavg_weights(weight_list, data_sizes)

    def broadcast(self, global_weights: Dict[str, np.ndarray]) -> None:
        for node in self.nodes:
            node.set_weights(deepcopy(global_weights))

    def _scaffold_init(self, global_w: Dict[str, np.ndarray]) -> None:
        # Vòng đầu c = c_i = 0
        if self.c is None:
            self.c = {k: np.zeros_like(v, dtype=np.float32)
                      for k, v in global_w.items()}
        for n in self.nodes:
            if n.c_i is None:
                n.c_i = {k: np.zeros_like(v, dtype=np.float32)
                         for k, v in global_w.items()}

    def _scaffold_prepare(self) -> None:
        # Model đang ở global_w nên c_i⁺ = ∇L_i(w_global)
        for n in self.nodes:
            n.c_i_next = n.full_gradient()

    def _scaffold_commit(self) -> None:
        # c_i ← c_i⁺ ;  c ← Σ (n_i/N)·c_i
        total = float(sum(n.n_train for n in self.nodes)) or 1.0
        for n in self.nodes:
            n.c_i = n.c_i_next
        self.c = {
            k: sum((n.n_train / total) * n.c_i[k] for n in self.nodes).astype(np.float32)
            for k in self.c
        }

    def run(self, fl_rounds: int = FL_ROUNDS,
            local_epochs: int = FL_LOCAL_EPOCHS,
            verbose: bool = True) -> List[dict]:
        """
        Mỗi vòng:
          1. Trung bình có trọng số → global_w, phát về các node
          2. Mỗi node train với gradient g - c_i + c
          3. Cập nhật control variate, đánh giá, ghi log
        """
        if verbose:
            print("        [SCAFFOLD] control variates, local optimizer = Adam")

        for r in range(1, fl_rounds + 1):
            global_w = self.aggregate()
            self.broadcast(global_w)

            self._scaffold_init(global_w)
            self._scaffold_prepare()
            for node in self.nodes:
                corr = {k: self.c[k] - node.c_i[k] for k in self.c}
                node.local_train(epochs=local_epochs, correction=corr)
            self._scaffold_commit()

            maes, rmses, mapes = [], [], []
            for node in self.nodes:
                mae, rmse, mape = node.evaluate()
                maes.append(mae); rmses.append(rmse); mapes.append(mape)

            avg_mae  = round(float(np.mean(maes)),  3)
            avg_rmse = round(float(np.mean(rmses)), 3)
            avg_mape = round(float(np.mean(mapes)), 3)

            self.round_log.append({
                "round":    r,
                "avg_MAE":  avg_mae,
                "avg_RMSE": avg_rmse,
                "avg_MAPE": avg_mape,
            })

            if verbose:
                print(f"        FL Round {r:2d}/{fl_rounds}: "
                      f"MAE={avg_mae:.2f} km/h  "
                      f"RMSE={avg_rmse:.2f}  "
                      f"MAPE={avg_mape:.1f}%")

        return self.round_log


def _train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                 epochs: int = FL_LOCAL_EPOCHS, lr: float = None,
                 correction: Optional[Dict[str, np.ndarray]] = None
                 ) -> None:
    """
    Train model bằng Adam, full-batch.
    correction (c - c_i) được truyền → SCAFFOLD: grad += correction trước optimizer.step().
    Không truyền (train cục bộ ban đầu) → train thường.
    """
    from config import LEARNING_RATE
    if lr is None:
        lr = LEARNING_RATE

    if not HAS_TORCH:
        if hasattr(model, "train_sgd"):
            model.train_sgd(X_train, y_train, epochs=epochs, lr=lr)
        return

    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    corr_t = None
    if correction is not None:
        corr_t = [(p, torch.tensor(correction[name], dtype=p.dtype))
                  for name, p in model.named_parameters()]

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        if corr_t is not None:
            for p, c in corr_t:
                if p.grad is not None:
                    p.grad.add_(c)
        optimizer.step()


def _full_gradient(model: Any, X_train: np.ndarray,
                   y_train: np.ndarray) -> Dict[str, np.ndarray]:
    """∇ MSE full-batch tại weights hiện tại → {tên_param: ndarray}."""
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    model.train()
    model.zero_grad()
    loss = nn.MSELoss()(model(X_t), y_t)
    loss.backward()
    grads = {name: (p.grad.detach().clone().cpu().numpy()
                    if p.grad is not None else np.zeros(p.shape, np.float32))
             for name, p in model.named_parameters()}
    model.zero_grad()
    return grads


def run_federated(
    method: str,
    node_data: Dict[str, dict],
    fl_rounds:    int = FL_ROUNDS,
    local_epochs: int = FL_LOCAL_EPOCHS,
    verbose:      bool = True,
) -> Tuple[List[dict], List[dict]]:
    """Chạy FL SCAFFOLD cho 1 session × 1 phương pháp."""
    if not HAS_TORCH:
        raise RuntimeError("SCAFFOLD cần PyTorch → pip install torch")

    nodes: List[FederatedNode] = []
    for seg, d in node_data.items():
        nodes.append(FederatedNode(
            node_id = seg,
            model   = d["model"],
            X_train = d["X_tr"],
            y_train = d["y_tr"],
            X_test  = d["X_te"],
            y_test  = d["y_te"],
            mn      = d["mn"],
            mx      = d["mx"],
        ))

    if not nodes:
        return [], []

    server    = FederatedServer(nodes)
    round_log = server.run(fl_rounds=fl_rounds, local_epochs=local_epochs,
                           verbose=verbose)

    fl_metrics = []
    for node in nodes:
        mae, rmse, mape = node.evaluate()
        local_mae  = node_data[node.node_id].get("local_mae",  0)
        local_rmse = node_data[node.node_id].get("local_rmse", 0)
        fl_metrics.append({
            "method":      method,
            "fl_algo":     "scaffold",
            "segment":     node.node_id,
            "MAE_local":   local_mae,
            "RMSE_local":  local_rmse,
            "MAE_fl":      mae,
            "RMSE_fl":     rmse,
            "MAPE_fl":     mape,
            "delta_MAE":   round(local_mae - mae, 3),
        })

    return fl_metrics, round_log
