from csv import writer
from glob import glob
import os

import MeCab
from nltk import ngrams
import pandas as pd

TEXT_PATH = ""

def aggregate_label(
    dir_path: str="/home/taku/FlareFormer/Flareformer/data_forecast/defn_labels/", 
    save_path: str="/home/taku/FlareFormer/Flareformer/data_forecast/defn_labels/all_predicts.txt",
    kind: str="predict"):

    # 1つの.txtファイルにまとめる
    text = ''
    all_label_path = glob(dir_path+f'*{kind}.txt')
    assert len(all_label_path) > 0

    print(f'Num {kind} text file >>> {len(all_label_path)}')
    for path in all_label_path:
        with open(path, 'r') as f:
            line = f.read()
        if len(line) == 0:
            continue
        else:
            text += line
            text += '\n'
    # 最後の改行を除く
    text = text[:-1]

    print(f'Save path >>> {save_path}')
    with open(save_path, 'w', encoding="UTF-8") as f:
        f.write(text)
    
    return save_path

def clean_labels(text: str):
    """
    ラベルが""に連続しているので、分割しやすいように置換する
    "ABC。""DEF。""GHI" -> ABC。 DEF。 GHI --> ['ABC。', 'DEF。', 'GHI']
    """
    text = text.replace('\"\"', ' ')
    text = text.replace('\"', '')
    text = text.replace('\\n', '') # TODO : \nをどうするか微妙
    text = text.split(' ')
    return text

def ngram2csv(ngram: dict, path, kind: str="uni"):
    """
    {'pattern', 'count'}の辞書を受け取り、
    'pattern', 'count'のcsvファイルとして出力する

    n-gramに変形する時にタプルになって見づらいので変換
    """
    print(f"Start {kind}_gram to csv")
    if kind == 'uni':
        ngram = sorted(ngram.items(), key=lambda x:x[1], reverse=True)
        for data in ngram:
            pattern = data[0][0]
            count = data[1]
            row = [pattern, count]
            with open(path, 'a') as f:
                w = writer(f)
                w.writerow(row)
    elif kind == "bi":
        ngram = sorted(ngram.items(), key=lambda x:x[1], reverse=True)
        for data in ngram:
            pattern = data[0]
            count = data[1]
            row = [pattern, count]
            with open(path, 'a') as f:
                w = writer(f)
                w.writerow(row)
    elif kind == "tri":
        ngram = sorted(ngram.items(), key=lambda x:x[1], reverse=True)
        for data in ngram:
            pattern = data[0]
            count = data[1]
            row = [pattern, count]
            with open(path, 'a') as f:
                w = writer(f)
                w.writerow(row)
    print('Finished')

def main():
    """
    """
    label_path = "/home/taku/FlareFormer/Flareformer/data_forecast/defn_labels/all_predicts.txt"
    if not os.path.exists(label_path):
        aggregate_label()
    
    with open(label_path, 'r') as f:
        labels = f.read()

    mecab = MeCab.Tagger('-Owakati')
    
    uni_grams = {}
    labels = labels.split('\n')
    for label in labels:
        data = mecab.parse(label)
        data = data.split()
        uni_gram = list(ngrams(data, 1))
        for uni in uni_gram:
            if uni in uni_grams:
                uni_grams[uni] += 1
            else:
                uni_grams[uni] = 1
    ngram2csv(uni_grams, "/home/taku/FlareFormer/Flareformer/data_forecast/unigram.csv", "uni")

    # bi-gramの生成
    bi_grams = {}
    for label in labels:
        data = mecab.parse(label)
        data = data.split()
        bi_gram = list(ngrams(data, 2))
        for bi in bi_gram:
            if bi in bi_grams:
                bi_grams[bi] += 1
            else:
                bi_grams[bi] = 1
    ngram2csv(bi_grams, "/home/taku/FlareFormer/Flareformer/data_forecast/bigram.csv", "bi")

    print('----- bi_gramの先頭5行を表示 -----')
    data = pd.read_csv("/home/taku/FlareFormer/Flareformer/data_forecast/bigram.csv", header=None)
    print(data.head())

    tri_grams = {}
    for label in labels:
        data = mecab.parse(label)
        data = data.split()
        tri_gram = list(ngrams(data, 3))
        for tri in tri_gram:
            if tri in tri_grams:
                tri_grams[tri] += 1
            else:
                tri_grams[tri] = 1
    ngram2csv(tri_grams, '/home/taku/FlareFormer/Flareformer/data_forecast/trigram.csv', 'tri')

    print('----- tri_gramの先頭5行を表示 -----')
    data = pd.read_csv('/home/taku/FlareFormer/Flareformer/data_forecast/trigram.csv', header=None)
    print(data.head())


def main_origin():
    """
    事前に↓コマンドでtxtファイルを作成しておく
    shuf ~/Desktop/CRT/ja_crt_ws_for_parser/datasets/Flare-B-set-midnight/labels/ja.ort.jsonl | jq 'recurse | select(.instructions?) | .instructions' | tr -d '[' | tr -d ']' | tr '\n' ' '| tr -d ' ' > labels_all.txt
    """
    label_path = '/home/initial/Desktop/CRT/ja_crt_ws_for_parser/japanese_crt/labels_all.txt'
    
    with open(label_path, 'r') as f:
        labels_all = f.read()
        labels = clean_labels(labels_all)
    
    m = MeCab.Tagger('-Owakati')
    # uni-gramの生成
    uni_grams = {}
    for label in labels:
        data = m.parse(label)
        data = data.split()
        uni_gram = list(ngrams(data, 1))
        for uni in uni_gram:
            if uni in uni_grams:
                uni_grams[uni] += 1
            else:
                uni_grams[uni] = 1
    ngram2csv(uni_grams, "unigram.csv", "uni")

    # bi-gramの生成
    bi_grams = {}
    for label in labels:
        data = m.parse(label)
        data = data.split()
        bi_gram = list(ngrams(data, 2))
        for bi in bi_gram:
            if bi in bi_grams:
                bi_grams[bi] += 1
            else:
                bi_grams[bi] = 1
    ngram2csv(bi_grams, "bigram.csv", "bi")

    print('----- bi_gramの先頭5行を表示 -----')
    data = pd.read_csv("bigram.csv", header=None)
    print(data.head())

    tri_grams = {}
    for label in labels:
        data = m.parse(label)
        data = data.split()
        tri_gram = list(ngrams(data, 3))
        for tri in tri_gram:
            if tri in tri_grams:
                tri_grams[tri] += 1
            else:
                tri_grams[tri] = 1
    ngram2csv(tri_grams, 'trigram.csv', 'tri')

    print('----- tri_gramの先頭5行を表示 -----')
    data = pd.read_csv('trigram.csv', header=None)
    print(data.head())


if __name__ == "__main__":
    main()