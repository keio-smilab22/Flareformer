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

import threading
import copy

class CallbackServer:
    """一発打ち・画像取得のためのコールバックを定義し、サーバを起動するクラス"""

    GET_IMAGE_LEN = 24  # 予測結果が表している時間の幅

    def __init__(self, args):
        with open(args.server_params, "r") as f:
            server_params = json.load(f)

        # ft_databaseを全件読み込み、timeをキーとした辞書に格納する
        self.date_dic = {}
        with open(server_params["ft_database_path"], "r") as f:
            for line in f.readlines():
                line_data = json.loads(line)
                str_date = self.to_str_key(datetime.datetime.strptime(line_data["time"], "%d-%b-%Y %H"))
                self.date_dic[str_date] = line_data

        self.host = server_params["hostname"]
        self.port = server_params["port"]

        # 一度のインファレンスで入力するデータ数。4であれば、4時間分のデータを入力する
        self.data_window_len = args.dataset["window"]

        # Lockのインスタンスを作成する
        self.tlock = threading.Lock()

    @staticmethod
    def parse_iso_time(iso_date):
        """ISO8601拡張形式の文字列をdatetime型に変換する"""
        parsed_date = datetime.datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return parsed_date

    @staticmethod
    def to_str_key(datetime_date):
        """datetime型の日時を指定フォーマとの文字列に変換する"""
        return datetime_date.strftime("%Y-%m-%d-%H")

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

    def make_targets_list(self, target_date_list):
        """データセットとターゲット日時のリストが合致するデータをリストに格納し返却する"""
        targets = []
        for target_date in target_date_list:
            if self.to_str_key(target_date) in self.date_dic:
                targets.append(self.date_dic[self.to_str_key(target_date)])
        return targets

    def start_server(self, callback):
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
            imgs = torch.cat([self.get_tensor_image(io.file.read()) for io in image_feats])
            phys = np.array([list(map(float, raw.split(","))) for raw in physical_feats])[:, :90]
            print(imgs.shape)
            prob = callback(imgs, phys).tolist()
            return JSONResponse(content={"probability": {"OCMX"[i]: prob[i] for i in range(len(prob))}})

        @fapi.get("/oneshot/simple", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot_simple(date: str):
            f_date = self.parse_iso_time(date)

            # カレンダーで指定した日時を含めて時刻を遡り、一度のインファレンスで入力する時刻をtarget_date_listに格納する
            target_date_list = []
            for offset in range(self.data_window_len):
                calc_date = f_date - datetime.timedelta(hours=offset)
                target_date_list.append(calc_date)
            target_date_list.reverse()

            # target_date_listと合致するデータをtargetsに格納する
            targets = self.make_targets_list(target_date_list)

            # 一発打ちに用いるデータが足りていない場合、failedとする
            if len(targets) != self.data_window_len:
                return JSONResponse(content={"probability": {"OCMX": []}, "oneshot_status": "failed"})

            # 一発打ちを実行する
            imgs = torch.cat([self.get_tensor_image_from_path(t["magnetogram"]) for t in targets])
            phys = np.array([list(map(float, t["feature"].split(","))) for t in targets])[:, :90]

            # 排他ロックをかける
            with self.tlock:
                prob = callback(imgs, phys).tolist()
                prob_cp = copy.deepcopy(prob)

            return JSONResponse(
                content={"probability": {"OCMX"[i]: prob_cp[i] for i in range(len(prob_cp))}, "oneshot_status": "success"}
            )

        @fapi.get("/images/path", responses={200: {"content": {"application/json": {"example": {}}}}})
        async def get_images_path(date: str):
            f_date = self.parse_iso_time(date)

            # カレンダーで指定した日時を含めずに1時間毎に時刻を取得し、計24時間分をtarget_date_listに格納する
            target_date_list = []
            for offset in range(self.GET_IMAGE_LEN):
                calc_date = f_date + datetime.timedelta(hours=offset + 1)
                target_date_list.append(calc_date)

            # target_date_listと合致するデータをtargetsに格納する
            targets = self.make_targets_list(target_date_list)

            # 合致した件数によってstatusを決定する
            if len(targets) == self.GET_IMAGE_LEN:
                status = "success"
            elif len(targets) == 0:
                status = "failed"
            else:
                status = "warning"

            # 画像パスのリストを作成する
            paths = [t["magnetogram"] for t in targets]

            return JSONResponse(content={"images": paths, "get_image_status": status})

        @fapi.get("/images/bin", response_class=FileResponse)
        async def get_images_bin(path: str):
            path = os.path.join(os.getcwd(), path)
            return path

        uvicorn.run(fapi, host=self.host, port=self.port)
