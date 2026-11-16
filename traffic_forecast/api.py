"""
api.py — FastAPI phục vụ dự báo giao thông (BiLSTM + SCAFFOLD)

Chạy:   uvicorn api:app --host 0.0.0.0 --port 8000
Docs:   http://localhost:8000/docs   (thử API trực tiếp trên trình duyệt)

Endpoint:
  GET  /health
  GET  /meta                       buổi, ngày (test/train), đoạn đường
  GET  /forecast?session=sang&date=20261102&segment=...&horizons=5,10,15,20,25,30
                                   [&origin=06:40]
  POST /reload                     nạp lại model/dữ liệu (sau khi thay saved_models)
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecaster

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    forecaster.load_all()          # nạp model + dữ liệu đúng 1 lần
    yield


app = FastAPI(title="Traffic Forecast API", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "sessions_loaded": sorted(forecaster.STATE.keys())}


@app.get("/meta")
def get_meta():
    return forecaster.meta()


@app.get("/forecast")
def get_forecast(
    session: str = Query(..., description="sang | trua | chieu"),
    date: str = Query(..., description="YYYYMMDD, lấy từ /meta"),
    segment: str = Query(..., description="tên đoạn đường, lấy từ /meta"),
    horizons: str = Query("5,10,15,20,25,30", description="các mốc phút, cách nhau bằng dấu phẩy"),
    origin: Optional[str] = Query(None, description="HH:MM, chỉ lấy 1 giờ gốc (vd 06:40)"),
):
    try:
        hs = [int(x) for x in horizons.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "horizons phải là danh sách số nguyên, ví dụ 5,10,15")
    try:
        return forecaster.forecast(session, date, segment, hs, origin)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))


@app.post("/reload")
def reload_models():
    forecaster.load_all()
    return {"status": "reloaded", "sessions_loaded": sorted(forecaster.STATE.keys())}


# Trang demo (static/index.html). Đặt CUỐI FILE: mount "/" đặt trước sẽ nuốt các route trên.
_static = os.path.join(BASE_DIR, "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
