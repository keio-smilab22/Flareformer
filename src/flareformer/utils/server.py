"""
Callback server
"""
import datetime
import json
import os
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.middleware.cors import CORSMiddleware
from torchvision import transforms


class CallbackServer:
    """一発打ち・画像取得のためのコールバックを定義し、サーバを起動するクラス"""

    @staticmethod
    def parse_iso_time(iso_date):
        """ISO8601拡張形式の文字列をdatetime型に変換する"""
        parsed_date = datetime.datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return parsed_date

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

    @classmethod
    def start_server(cls, callback):
        """
        Start http server.
        """
        fapi = FastAPI()
        fapi.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
        )
        params = json.loads(open("config/params_2017.json").read())
        feature_len = params['dataset']['window']

        @fapi.post("/oneshot/full", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot_full(
            image_feats: List[UploadFile] = File(description="four magnetogram images"),
            physical_feats: List[str] = Form(description="physical features"),
        ):
            imgs = torch.cat([cls.get_tensor_image(io.file.read()) for io in image_feats])
            phys = np.array([list(map(float, raw.split(","))) for raw in physical_feats])[:, :90]
            print(imgs.shape)
            prob = callback(imgs, phys).tolist()
            return JSONResponse(content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}})

        @fapi.get("/oneshot/simple", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot_simple(date: str):
            f_date = cls.parse_iso_time(date)
            query_date = f_date.strftime("%Y-%m-%d-%H")
            jsonl_database_path = "data/ft_database_all17.jsonl"
            targets = []
            prob = []
            status = "failed"
            with open(jsonl_database_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    targets.append(data)
                    if len(targets) > feature_len:
                        targets.pop(0)
                    target_date = datetime.datetime.strptime(data["time"], "%d-%b-%Y %H").strftime("%Y-%m-%d-%H")
                    if query_date == target_date:
                        imgs = torch.cat(
                            [cls.get_tensor_image_from_path(t["magnetogram"]) for t in targets]
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
        def execute_images_path(date: str):
            f_date = cls.parse_iso_time(date)
            query_date = f_date.strftime("%Y-%m-%d-%H")
            jsonl_database_path = "data/ft_database_all17.jsonl"
            finish_date = (f_date + datetime.timedelta(hours=24)).strftime("%Y-%m-%d-%H")
            targets = []
            with open(jsonl_database_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    target_date = datetime.datetime.strptime(data["time"], "%d-%b-%Y %H").strftime("%Y-%m-%d-%H")
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
        def execute_images_bin(path: str):
            path = os.path.join(os.getcwd(), path)
            return path

        host_name = "0.0.0.0"
        port_num = 8080
        uvicorn.run(fapi, host=host_name, port=port_num)
