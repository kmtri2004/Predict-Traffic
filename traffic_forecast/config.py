import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

SESSION_MAP = {"S": "sang", "T": "trua", "C": "chieu"}

SESSION_LABEL = {
    "sang": "Buổi sáng (6:30–7:30)",
    "trua": "Buổi trưa (12:00–13:00)",
    "chieu": "Buổi chiều (17:00–18:00)",
}

SEQ_LEN = 3
STEP_MINUTES = 5          # khoảng cách giữa 2 snapshot liên tiếp
HORIZONS = [5, 15, 30, 60]  # các mốc dự báo (phút) mặc định
TRAIN_RATIO = 0.8

HIDDEN_SIZE = 32
NUM_LAYERS = 1
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 30
BATCH_SIZE = 16

FL_ROUNDS = 15
FL_LOCAL_EPOCHS = 5

FL_ALGO = "scaffold"

METRICS = ["MAE", "RMSE", "MAPE"]

RANDOM_SEED = 42


def _algo_tag() -> str:
    # "_byday": chuỗi trượt theo từng ngày -> tách thư mục, không nạp nhầm model cũ
    return f"_{FL_ALGO}_byday"


def comparison_csv() -> str:
    return os.path.join(OUTPUT_DIR, f"comparison_all{_algo_tag()}.csv")


def out_dir(session_key: str, method: str) -> str:
    path = os.path.join(OUTPUT_DIR, session_key, method + _algo_tag())
    os.makedirs(path, exist_ok=True)
    return path


def session_out_dir(session_key: str) -> str:
    path = os.path.join(OUTPUT_DIR, session_key)
    os.makedirs(path, exist_ok=True)
    return path


def model_save_dir(session_key: str, method: str) -> str:
    path = os.path.join(MODEL_DIR, session_key, method + _algo_tag())
    os.makedirs(path, exist_ok=True)
    return path
