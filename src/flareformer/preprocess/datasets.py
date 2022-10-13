"""Datasets for preprocess"""
import json
from dataclasses import dataclass
from typing import List

from numpy import ndarray
from PIL import Image
from torchvision import transforms


@dataclass
class YearSection:
    """
    Year section class
    """

    start: int
    end: int


def read_jsonl(path: str) -> List[dict]:
    """
    josnlファイルをdictの配列として読み込む
    """
    with open(path) as f:
        return [json.loads(line) for line in f]


def detect_year_sections(jsonl: List[dict]):
    """
    Detect year sections
    """
    secs = {}
    for i, data in enumerate(jsonl):
        year = int(data["time"][-7:-3])
        if year not in secs:
            secs[year] = YearSection(start=1e18, end=-1)
        secs[year].start = min(secs[year].start, i)
        secs[year].end = max(secs[year].end, i)

    years = sorted(secs.keys())
    return secs, years


def split_dataset(jsonl: List[dict], sections: dict) -> dict:
    """
    dictの配列データを純粋な辞書形式に変換
    """
    results = {}
    for k, v in sections.items():
        results[k] = jsonl[v.start : v.end + 1]
    return results


def get_image(img_path: str, resize_size: int = 512) -> ndarray:
    """
    指定されたパスの画像ファイルをnumpy形式で読み込む
    """
    transform = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform(img)
    img = img[0, :, :].unsqueeze(0)
    return img.cpu().numpy()
