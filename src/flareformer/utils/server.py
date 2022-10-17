"""
Callback server
"""
import datetime
import json
import locale
import os
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from torchvision import transforms


class Date(BaseModel):
    """Date class"""

    year: str
    month: str
    day: str
    hour: str


class CallbackServer:
    """一発打ち・画像取得のためのコールバックを定義し、サーバを起動するクラス"""

    @staticmethod
    def get_tensor_image(img_buff):
        """画像のRAWデータをTensorに変換"""
        transform = transforms.Compose([transforms.ToTensor()])
        img = Image.frombytes(mode="RGB", size=(256, 256), data=img_buff)
        img = transform(img)
        img = img[0, :, :].unsqueeze(0)
        return img.unsqueeze(0)

    @staticmethod
    def get_tensor_image_from_path(path):
        """指定されたパスの画像ファイルを読み込んでTensor形式で返す"""
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        img_pil = Image.open(path)
        img = transform(img_pil)[0, :, :].unsqueeze(0)
        return img.unsqueeze(0)

    @staticmethod
    def start_server(callback):
        """
        Start http server.
        """
        fapi = FastAPI()
        fapi.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
        )

        @fapi.post("/oneshot/full", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot_full(
            image_feats: List[UploadFile] = File(description="four magnetogram images"),
            physical_feats: List[str] = Form(description="physical features"),
        ):
            imgs = torch.cat([CallbackServer.get_tensor_image(io.file.read()) for io in image_feats])
            phys = np.array([list(map(float, raw.split(","))) for raw in physical_feats])[:, :90]
            print(imgs.shape)
            prob = callback(imgs, phys).tolist()
            return JSONResponse(content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}})

        @fapi.get("/oneshot/simple", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot_simple(date: str):
            # ISO8601の基本形式から拡張形式へ変換
            s_date = datetime.datetime.strptime(date, '%Y%m%dT%H%M%S').isoformat()
            # datetime型に変換する
            f_date = datetime.datetime.fromisoformat(s_date)
            query_date = f_date.strftime("%Y-%m-%d-%H")
            jsonl_database_path = "data/ft_database_all17.jsonl"
            targets = []
            prob = []
            status = "failed"
            with open(jsonl_database_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    targets.append(data)
                    if len(targets) > 4:
                        targets.pop(0)
                    target_date = datetime.datetime.strptime(data["time"], "%d-%b-%Y %H").strftime("%Y-%m-%d-%H")
                    if query_date == target_date:
                        imgs = torch.cat(
                            [CallbackServer.get_tensor_image_from_path(t["magnetogram"]) for t in targets]
                        )
                        phys = np.array([list(map(float, t["feature"].split(","))) for t in targets])[:, :90]
                        print(imgs.shape)
                        prob = callback(imgs, phys).tolist()
                        status = "success"
                        break

            return JSONResponse(
                content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}, "oneshot_status": status}
            )

        @fapi.get("/images/path", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot_images_path(date: str):
            # ISO8601の基本形式から拡張形式へ変換
            s_date = datetime.datetime.strptime(date, '%Y%m%dT%H%M%S').isoformat()
            # datetime型に変換する
            f_date = datetime.datetime.fromisoformat(s_date)
            query_date = f_date.strftime("%Y-%m-%d-%H")
            jsonl_database_path = "data/ft_database_all17.jsonl"
            finish_date = query_date + datetime.timedelta(hours=24)
            targets = []
            with open(jsonl_database_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    target_date = datetime.datetime.strptime(data["time"], "%d-%b-%Y %H")
                    if query_date < target_date <= finish_date:
                        targets.append(data)

            if len(targets) == 0:
                status = "failed"
            elif len(targets) < 24:
                status = "warning"
            else:
                status = "success"

            paths = [t["magnetogram"] for t in targets]
            return JSONResponse(content={"images": paths, "get_image_status": status})

        @fapi.get("/images/bin", response_class=FileResponse)
        def execute_oneshot_images_bin(path: str):
            path = os.path.join(os.getcwd(), path)
            return path

        host_name = "0.0.0.0"
        port_num = 8080
        uvicorn.run(fapi, host=host_name, port=port_num)
