"""Preprocess utils"""
import datetime


def make_prefix(output_path: str, year: int, data_type: str) -> str:
    """
    Make proper prefix of path
    """
    return f"{output_path}/data_{year}_{data_type}"


def identity(_x):
    """
    Identity
    """
    return _x


def get_time(str_time: str):
    """
    Get time from string
    """
    return datetime.datetime.strptime(str_time, '%d-%b-%Y %H')


def sub_1h(str_time):
    """
    Substract 1 hour from time
    """
    time = get_time(str_time)
    next_time = time - datetime.timedelta(hours=1)
    next_time = datetime.datetime.strftime(next_time, '%d-%b-%Y %H')
    return next_time
