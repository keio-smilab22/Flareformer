"""
Callback server
"""
import uvicorn
import torch
import json
import datetime
import locale
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from typing import List
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel


class Date(BaseModel):
    year: str
    month: str
    day: str
    hour: str


class CallbackServer:
    @staticmethod
    def get_tensor_image(img_buff):
        transform = transforms.Compose([transforms.ToTensor()])
        img = Image.frombytes(mode="RGB", size=(256, 256), data=img_buff)
        img = transform(img)
        img = img[0, :, :].unsqueeze(0)
        return img.unsqueeze(0)

    @staticmethod
    def get_tensor_image_from_path(path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_pil = Image.open(path)
        img = transform(img_pil)[0, :, :].unsqueeze(0)
        return img.unsqueeze(0)

    @staticmethod
    def start(callback):
        """
        Function of start http server
        """
        fapi = FastAPI()
        fapi.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        @fapi.post(
            "/",
            responses={
                200: {
                    "content": {"application/json": {"example": {}}}
                }
            }
        )
        def execute_oneshot(
            image_feats: List[UploadFile] = File(description="four magnetogram images"),
            physical_feats: List[str] = Form(description="physical features")
        ):
            imgs = torch.cat([CallbackServer.get_tensor_image(io.file.read()) for io in image_feats])
            phys = np.array([list(map(float, raw.split(","))) for raw in physical_feats])[:, :90]
            print(imgs.shape)
            prob = callback(imgs, phys).tolist()
            return JSONResponse(content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}})

        @fapi.post(
            "/simple",
            responses={
                200: {
                    "content": {"application/json": {"example": {}}}
                }
            }
        )
        def execute_oneshot(
            date: Date
        ):
            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
            jsonl_database_path = "data/ft_database_all17.jsonl"
            query = f"{date.year}-{date.month}-{date.day}-{date.hour}"
            targets = []
            with open(jsonl_database_path, "r") as f:
                for line in f.readlines():
                    data = json.loads(line)
                    targets.append(data)
                    if len(targets) > 4:
                        targets.pop(0)
                    query_date = datetime.datetime.strptime(query, '%Y-%m-%d-%H').strftime('%Y-%m-%d-%H')
                    target_date = datetime.datetime.strptime(data["time"], '%d-%b-%Y %H').strftime('%Y-%m-%d-%H')
                    if query_date == target_date:
                        break

            imgs = torch.cat([CallbackServer.get_tensor_image_from_path(t["magnetogram"]) for t in targets])
            phys = np.array([list(map(float, t["feature"].split(","))) for t in targets])[:, :90]

            print(imgs.shape)
            prob = callback(imgs, phys).tolist()
            return JSONResponse(content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}})

        host_name = "127.0.0.1"
        port_num = 8080
        uvicorn.run(fapi, host=host_name, port=port_num)
