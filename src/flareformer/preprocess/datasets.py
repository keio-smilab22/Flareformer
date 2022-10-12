"""Datasets for preprocess"""
import json
from typing import List
from dataclasses import dataclass
from numpy import ndarray
from torchvision import transforms
from PIL import Image


@dataclass
class YearSection:
    """
    Year section class
    """
    start: int
    end: int


def read_jsonl(path: str) -> List[dict]:
    """
    Read jsonl
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
    Split dataset
    """
    results = {}
    for k, v in sections.items():
        results[k] = jsonl[v.start:v.end + 1]
    return results


def get_image(img_path: str, resize_size: int = 512) -> ndarray:
    """
    Get image
    """
    transform = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform(img)
    img = img[0, :, :].unsqueeze(0)
    return img.cpu().numpy()
