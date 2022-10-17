#! /usr/bin/env python
"""
太陽フレア予報文をパースして、必要部分を取り出す
"""
"""
年度データ名, parse前, parse後 をcsvファイルに保存
if の直下
    line_or = line
を追加

if 最後
with open('./HisParser/his_XXX.csv', 'a') as f:
                        print(args.input.name.split('/')[-1], end=",", file=f)
                        print(line_or, end=",", file=f)
                        print(line, file=f)

"""

import argparse
from multiprocessing import context
import os
import re


def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=argparse.FileType('r'))
    parser.add_argument('--output', '-o')
    parser.add_argument('--rm_solar_wind', action="store_true")
    args = parser.parse_args()
    return args


def divide_datetime(date_time: str):
    '''
    Args:
        data_time : XXXX年YY月ZZ日
    Return:
        year : XXXX
        month : YY
        day : ZZ
    '''
    year, date_time = date_time.split('年')
    month, date_time = date_time.split('月')
    day = date_time.split('日')[0]

    return int(year), int(month), int(day)


def clean_date_description(day, line):
    """
    Args:
        line (str): parser対象の文字列
    
    (1) XX~YY日 -> XX日
    (2) (XX月)?XX日(XX時)?(XX分)?(UT)?現在、 -> 現在、
    (3) (XX月)?XX日(XX時)?(XX分)?(UT)? -> XX日XX時

    (4) 日付表記 -> 先日 / 昨日 / 本日 / 明日 / 明日以降

    (5) XX時頃?に? -> 本日
        XX時の情報しかもたないものがあるため。
    
    """
    if re.search(r'〜[０-９]{1,2}日', line):
        line = re.sub(r'〜[０-９]{1,2}', r'', line)
    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?現在、', line):
        line = re.sub(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?現在、', r'現在、', line)
    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
        line = re.sub(r'([０-９]{1,2}月)?', r'', line)
        line = re.sub(r'([０-９]{1,2}分)?(（ＵＴ）)?', r'', line)
    
    line = change_day_description(day, line)
    
    if re.search(r'[０-９]{1,2}時頃?に?', line):
        line = re.sub(r'[０-９]{1,2}時頃?に?', r'本日', line)
    
    return line


def change_day_description(day, text):
    '''
    textに含まれるXX日を
        先日・昨日・本日・明日・明日以降
    に変換する

    trans_table = str.maketrans({"a":"A", "d":"D"})
    text.translate(trans_table)
    '''
    zen2han = str.maketrans({'０':'0', '１':'1', '２':'2', '３':'3', '４':'4',
                                '５':'5', '６':'6', '７':'7', '８':'8', '９':'9'})
    han2zen = str.maketrans({'0':'０', '1':'１', '2':'２', '3':'３', '4':'４',
                                '5':'５', '6':'６', '7':'７', '8':'８', '9':'９'})

    text = text.translate(zen2han)

    def change_day(day, idx, text, search_text):
        dif = idx - day

        # day = 1, 2 のときかつ idx=28,29,30,31の時が
        if day==1 and idx in [28, 29, 30, 31]:
            text = re.sub(search_text, '昨日以前', text)
        elif day==2 and idx in [28, 29, 30, 31]:
            text = re.sub(search_text, '先日', text)
        
        # day = 27, 28, 29, 30, 31のときidx=1,2,3の時
        # 27日->2日までが存在
        if day == 27 and idx in [1,2,3]:
            text = re.sub(search_text, '明日以降', text)
        if day == 28 and idx in [1,2,3]:
            text = re.sub(search_text, '明日以降', text)
        if day == 29 and idx in [1,2,3]:
            text = re.sub(search_text, '明日以降', text)
        if day == 30 and idx in [1,2,3]:
            text = re.sub(search_text, '明日以降', text)
        if day == 31 and idx in [1,2,3]:
            text = re.sub(search_text, '明日以降', text)
        
        elif dif < -1:
            text = re.sub(search_text, '先日', text)
        elif dif == -1:
            text = re.sub(search_text, '昨日', text)
        elif dif == 0:
            text = re.sub(search_text, '本日', text)
        elif dif == 1:
            text = re.sub(search_text, '明日', text)
        else:
            text = re.sub(search_text, '明日以降', text)
        
        return text

    def change_seq(day, idx, text, search_text):
        '''
        XXX日からXXX日にかけて --> 本日以降にかけて | 明日以降にかけて
        '''
        dif = idx - day

        if dif == 0:
            text = re.sub(search_text, '本日以降に', text)
        else:
            text = re.sub(search_text, "明日以降に", text)
        
        return text
    
    # XX日からYY日にかけてを先に処理
    for idx in range(day-5, day+5):
        search_text = str(idx) + '日' + r'([0-9]{1,2}時)?' + r'頃?' + 'から' + r'[0-9]{1,2}日([0-9]{1,2}時)?頃?にかけて'
        if re.search(search_text, text):
            text = change_seq(day, idx, text, search_text)
    
    # 日付表記を変換
    for idx in range(31, 0, -1):
        search_text = str(idx) + '日' + r'([0-9]{1,2}時)?' + r'頃?に?'
        if re.search(search_text, text):
            text = change_day(day, idx, text, search_text)
    
    # その後のparserのため日付表記以外を全角に戻す
    return text.translate(han2zen)


def clean_named_entity(line):
    '''
    (1) ＣＭＥ（コロナ質量放出） -> コロナ質量放出
    (2) ＳＣ型(（急始型）)?地磁気嵐 -> 急始型地磁気嵐
    (3) ＳＧ型(（緩始型）)?地磁気嵐 -> 緩始型地磁気嵐
    (4) ＳＩ(（地磁気水平成分の急増）)? -> 地磁気水平成分の急増
    (5) ベイ（地磁気水平成分の湾形変化） -> 地磁気水平成分の湾形変化
    (6) ＳＦＥ（フレアによるＸ線の影響で起こる地磁気変化） -> フレアによるＸ線の影響で起こる地磁気変化
    (7) 野辺山電波...によると、 -> 削除
    '''
    if re.search(r'ＣＭＥ(（コロナ質量放出）)?', line):
        line = re.sub(r'ＣＭＥ(（コロナ質量放出）)?', r'コロナ質量放出', line)
    if re.search(r'ＳＣ型(（急始型）)?', line):
        line = re.sub(r'ＳＣ型(（急始型）)?', r'急始型', line)
    if re.search(r'ＳＧ型(（緩始型）)?地磁気嵐', line):
        line = re.sub(r'ＳＧ型(（緩始型）)?', r'緩始型', line)
    if re.search(r'ＳＩ(（地磁気水平成分の急増）)?', line):
        line = re.sub(r'ＳＩ(（地磁気水平成分の急増）)?', r'地磁気水平成分の急増', line)
    if re.search(r'ベイ（地磁気水平成分の湾形変化）', line):
        line = re.sub(r'ベイ（地磁気水平成分の湾形変化）', r'地磁気水平成分の湾形変化', line)
    if re.search(r'ＳＦＥ（フレアによるＸ線の影響で起こる地磁気変化）', line):
        line = re.sub(r'ＳＦＥ（フレアによるＸ線の影響で起こる地磁気変化）', r'フレアによるＸ線の影響で起こる地磁気変化', line)
    if re.search(r'野辺山電波.*よると、', line):
        line = re.sub(r'野辺山電波.*よると、', r'', line)

    return line


def clean_nT(line):
    '''
    # -nT
    (1) -XXnT~-nTの -> 削除
    (2) -XXnT(前後|付近|程度)?(の|まで) -> 削除
    
    # nT
    (3) 急始型地磁気嵐による地磁気水平成分の変化量は約１０１ｎＴで、現在も~ 
        ==> 影響は に置換
    (4) XXXnT~XXnT前後 -> 状態 に置換
    (5) 現在XXnT前後の  -> 現在 に置換 
    (6) XXXnTからXXXnT
    (7) XXXnT(程度|付近|前後)?の -> 状態 に置換
    (8) XXXnTから      -> 状態 に置換
    (9) XXXnT(前後|付近|程度)?(、|へ|まで|を|で|に)? -> 状態 に置換

    (10) 状態の状態 -> 状態
    '''
    # -nT
    if re.search(r'−[０-９]{1,3}(ｎＴ)?(から|〜)−[０-９]{1,3}(ｎＴ)?の?', line):
        line =re.sub(r'−[０-９]{1,3}(ｎＴ)?から−[０-９]{1,3}(ｎＴ)?の?', r'', line)
    if re.search(r'−[０-９]{1,3}ｎＴ(前後|付近|程度)?(の|まで)', line):
        line = re.sub(r'−[０-９]{1,3}ｎＴ(前後|付近|程度)?(の|まで)', r'', line)
    
    # nT
    if re.search(r'地磁気.*変化量は.*[０-９]{1,3}ｎＴで', line):
        line = re.sub(r'地磁気.*変化量は.*[０-９]{1,3}ｎＴで', r'の影響は', line)
    if re.search(r'[０-９]{1,3}(ｎＴ)?(から|〜)[０-９]{1,3}ｎＴ前後?', line):
        line = re.sub(r'[０-９]{1,3}(ｎＴ)?(から|〜)[０-９]{1,3}ｎＴ前後?', r'状態', line)
    if re.search(r'現在[０-９]{1,3}ｎＴ(前後)?の?', line):
        line = re.sub(r'[０-９]{1,3}ｎＴ(前後)?の?', r'', line)
    if re.search(r'[０-９]{1,3}ｎＴから[０-９]{1,3}ｎＴの?間', line):
        line = re.sub(r'[０-９]{1,3}ｎＴから[０-９]{1,3}ｎＴの?間', r'状態', line)
    if re.search(r'[０-９]{1,3}〜[０-９]{1,3}ｎＴ(程度|付近|前後)?の', line):
        line = re.sub(r'[０-９]{1,3}〜[０-９]{1,3}ｎＴ(程度|付近|前後)?の', r'', line)
    if re.search(r'[０-９]{1,3}ｎＴから', line):
        line = re.sub(r'[０-９]{1,3}ｎＴから', r'状態', line)
    if re.search(r'[０-９]{1,3}ｎＴ(前後|付近|程度)?(、|へ|まで|を|で|に)?', line):
        line = re.sub(r'[０-９]{1,3}ｎＴ(前後|付近|程度)?', r'状態', line)
    
    if re.search(r'状態の状態', line):
        line = re.sub(r'状態の', r'', line)

    return line


def clean_velocity(line):
    """
    ｋｍ／ｓ
    (1) (低、高)速なXXX (km/s)?から、?XXX km/s(前後|付近)?(に|へ) -> AAな状態から に置換
    (2) (低、高)速なXXX (km/s)?から、?(やや)?(低、高)速なXXX km/s(前後|付近)?(に|へ)達し -> AAな状態からBBな状態へ に置換
    (3) (低、高)速なXXX -> 通常速度 の(2)のパターン
    (4) (低、高)速なXXX -> 通常速度 の(2)のパターン2
    (4) 通常速度 -> (低、高)速なXXX の(2)のパターン

    (2) (低、高)速なXXX (km/s)?から、?XXX km/s(前後|付近)?(で|を|に|へ|の間) -> (低、高)速な状態から に置換
    (3) (低、高)速なXXX km/s(前後|付近)?(に|へ|で)? -> (低、高)速な状態(に|へ|で)? に置換
    (4) 通常速度 の?XXXkm/s[前後] -> 通常速度の?[前後]
    (5) XXX~XXXkm/s -> 状態
    (6) XXX(km/s)?(程度)(~|から)XXXkm/s(付近)?(へと)? -> 削除
    (7) XXXkm/s(付近へ|付近の|程度の|前後の|前後で|前後を) -> 削除

    """
    # 前側にのみ高速or低速がある場合
    if re.search(r'(かなり|やや)?(低|高)速な[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(に|へ)', line):
        line = re.sub(r'[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(に|へ)', r'状態から', line)
    # 前後両方に高速or低速がある場合
    if re.search(r'(かなり|やや)?(低|高)速な[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?(一時)?(やや|かなり)?(高速な|低速な)?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(に|へ|の間)(上昇|達)し', line):
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近|程度)?から、?', r'状態から', line)
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|を|に|へ|の間)', r'状態へ', line)
    # 低|高速 -> 通常速度
    if re.search(r'(かなり|やや)?(低|高)速な[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?(一時)?通常速度の[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|に|へ|の間)', line):
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近|程度)?から、?', r'状態から', line)
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|を|に|へ|の間)', r'状態へ', line)
    # 低|高速 -> 通常速度 2
    if re.search(r'(かなり|やや)?(低|高)速な[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?の通常速度(まで|に|へ)', line):
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近|程度)?から、?', r'状態から', line)
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?の', r'', line)
    # 前側のみに通常速度がある場合
    if re.search(r'通常速度の?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?から、?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|に|へ|の間)', line):
        line = re.sub(r'の?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?から、?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?の間で?', r'の間で', line)
        line = re.sub(r'の?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?から、?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|に|へ)', r'から', line)
        if re.search(r'その後[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?(で|に|へ)', line):
            line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?(で|に|へ)', r'', line)
    # 通常速度 -> 低|高速
    if re.search(r'通常速度の[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?から、?(一時)?(やや|かなり)?(高速な|低速な)?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|に|へ|の間)', line):
        line = re.sub(r'の[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近|程度)?から、?', r'から', line)
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|を|に|へ|の間)', r'状態へ', line)
    # 通常速度 -> 低|高速 2
    if re.search(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?の通常速度から、?(一時)?(やや|かなり)?(高速な|低速な)?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(まで|で|に|へ|の間)', line):
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?の', r'', line)
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(まで|で|に|へ|の間)', r'状態へ', line)
    if re.search(r'(低|高)速な[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?(やや)?(高速な|低速な)?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|程度|付近)?(で|を|に|へ|の間)', line):
        line = re.sub(r'[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(前後|付近|程度)?から、?(やや)?(高速な|低速な)?[０-９]{1,3}[ａ-ｚ]{1,4}／[ａ-ｚ](前後|付近|程度)?(を|の間)?', r'状態', line)
    if re.search(r'(低|高)速な[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近)?(に|へ|で)?', line):
        line = re.sub(r'速な[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近)?', r'速な状態', line)
    if re.search(r'通常速度の?[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ]', line):
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ]', r'', line)
    if re.search(r'[０-９]{1,3}〜[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ]', line):
        line = re.sub(r'[０-９]{1,3}〜[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ]', r'状態', line)
    if re.search(r'[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(程度|前後|付近)?(〜|から)[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](付近|前後|程度)?(へと)?', line):
        line = re.sub(r'[０-９]{1,3}([ａ-ｚ]{1,2}／[ａ-ｚ])?(程度|前後|付近)?(〜|から)[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](付近|前後|程度)?(へと)?', r'', line)
    if re.search(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](付近へ|付近の|程度の|前後の|前後で|前後を)', line):
        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](付近へ|付近の|程度の|前後の|前後で|前後を)', r'', line)
    return line


def clean_angle(line: str):
    """
    (1) (東西南北)XX度(付近) -> (東西南北)側
    (2) 緯度のXX度付近? -> 緯度付近
    """
    if re.search(r'(東|西|南|北)[０-９]{1,3}度', line):
        line = re.sub(r'[０-９]{1,3}度', r'側', line)
    if re.search(r'(緯度|軽度)の[０-９]{1,3}度', line):
        line = re.sub(r'[０-９]{1,3}度', r'', line)
    
    return line


def clean_class(line: str):
    """
    (7) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
    (2) (X|M|B|C)Nクラス -> (X|M|B|C)クラス
    """
    # 小数点
    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?', line):
        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
    if re.search(r'[０-９]クラス', line):
        line = re.sub(r'[０-９]クラス', r'クラス', line)
    
    return line


def parse_forecast(
    input_path: str="/home/initial/Flareformer/data_forecast/defn_txt_2011-17/",
    output_path: str="/home/initial/Flareformer/data_forecast/flare_detail/"):
    '''
    Args:
        input_path : parser前のtextファイルへのpath
        output_path : 保存する先のpath
    '''
    with open(input_path, 'r') as f:
        text = f.read()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    str_summary = ""
    str_detail_sun = ""
    str_detail_geomagnetism = ""
    str_etc = ""
    text = text.replace('<br>','\n')
    md = re.split('----+', text)
    index = 0

    for text in md:
        if index == 0:
            #　日報：2017年09月18日　15時00分　>> XXX年/YY月/ZZ日 を抽出
            lines = text.split('\n')
            date_time = lines[0].split('：')[-1].split('\u3000')[0]
            year, month, day = divide_datetime(date_time)

        if index == 1:  # 概況
            info_flag = False
            lines = text.split('\n')
            for line in lines:
                if re.search(r'\t', line):
                    line = re.sub(r'\t', r'', line)
                
                if re.match(r'<font',line):
                    str_etc += line + "\n"
                elif re.match(r'^※',line):
                    str_etc += line + "\n"
                elif len(line) == 0:
                    str_etc += line + "\n"
                elif re.match(r'【お知らせ】', line):
                    info_flag = True
                elif re.match(r'【概況】', line):
                    info_flag = False
                elif info_flag:
                    pass
                elif re.search(r'臨時警報', line):
                    pass
                elif re.match(r".*高エネルギー電子", line):
                    pass
                elif re.match(r'^地磁気', line):
                    """
                    (1) 日付表現
                    """
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    str_summary += line + "\n"
                
                elif re.match(r'.*日間、地磁気.*', line):
                    str_summary += line + "\n"
                
                elif re.search(r'プロトン粒子フラックス', line):
                    pass
                
                elif re.search(r'ＧＬＥ：Ｇｒｏｕｎｄ　Ｌｅｖｅｌ　Ｅｖｅｎｔ', line):
                    pass
                
                elif re.search(r"プロトン..", line):
                    """
                    (1) 日付表現
                    (2) （※） -> 削除
                    """
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # その他　
                    if re.search(r'（※）', line):
                        line = re.sub(r'（※）', r'', line)
                    
                    str_summary += line + '\n'
                
                elif re.search(r'地磁気..', line):
                    """
                    (1) 角度表現
                    (3) 速度表現
                    (4) ｎＴ表現
                    (5) 日付表現
                    (6) 固有表現
                    """
                    # 角度表現
                    if re.search(r'[０-９]{1,3}度', line):
                        line = clean_angle(line)
                    # 速度表現
                    if re.search(r'([ａ-ｚ]{1,2}／[ａ-ｚ])', line):
                        line = clean_velocity(line)
                    # nT表現
                    if re.search(r'ｎＴ', line):
                        line = clean_nT(line)
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # 固有表現
                    line = clean_named_entity(line)
                    
                    str_summary += line + '\n'
                
                elif re.search(r"コロナホール", line):
                    pass
                    '''
                    (1) 角度表現
                    (2) ｎＴ表現
                    (3) 日付表現
                    (4) 固有表現
                    '''
                    # 角度表現
                    if re.search(r'[０-９]{1,3}度', line):
                        line = clean_angle(line)
                    # nT表現
                    if re.search(r'ｎＴ', line):
                        line = clean_nT(line)
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # 固有表現
                    line = clean_named_entity(line)
                    
                    str_summary += line + '\n'
                
                elif re.search(r".始型地?磁気嵐", line):
                    """
                    (1) 日付表現
                    """
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    
                    str_summary += line + "\n"
                
                elif re.search(r'継続中', line):
                    pass
                
                elif re.search(r"太陽風速度は?", line):
                    """
                    (1) 速度表現
                    (2) ｎＴ表現
                    (3) 日付表現
                    (4) 固有表現
                    (10) フレアクラス表現
                    """

                    # 速度表現
                    if re.search(r'([ａ-ｚ]{1,2}／[ａ-ｚ])', line):
                        line = clean_velocity(line)
                    # nT表現
                    if re.search(r'ｎＴ', line):
                        line = clean_nT(line)
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # 固有表現
                    line = clean_named_entity(line)
                    # フレアクラス表現
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?(フレア|クラス)?', line):
                        line = clean_class(line)
                    
                    str_summary += line + '\n'

                elif re.search(r'太陽風の?磁場', line):
                    """
                    (1) nT表記
                    (2) 日付表現
                    (3) 固有表現
                    """
                    # nT表現
                    if re.search(r'ｎＴ', line):
                        line = clean_nT(line)
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # 固有表現
                    line = clean_named_entity(line)

                    str_summary += line + "\n"

                elif re.search(r'ＣＭＥ', line):
                    """
                    (1) 日付表現
                    (2) フレアクラス表現
                    (3) 固有表現
                    """
                    # 日付表現
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # フレアクラス表現
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?(フレア|クラス)?', line):
                        line = clean_class(line)
                    # 固有表現
                    line = clean_named_entity(line)
                    
                    str_summary += line + "\n"

                elif re.search(r'(この)|(最大の)フレアは、', line):
                    """
                    (1) 活動領域XXXX -> 活動領域
                    (2) 日付表現 
                    (3) 太陽フレアクラス表現
                    (4) 固有表現
                    """
                    # 活動領域
                    if re.search(r'活動領域[０-９]{4}', line):
                        line = re.sub(r'活動領域[０-９]{4}', r'活動領域', line)
                    # 日付
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = clean_date_description(day, line)
                    # フレアクラス表現
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?(フレア|クラス)?', line):
                        line = clean_class(line)
                    # 固有表現
                    line = clean_named_entity(line)

                    str_summary += line + '\n'
                
                elif re.match(r'太陽面上?の?.*端で', line):
                    str_summary += line + '\n'
                
                elif re.match(r'太陽面では?', line):
                    """
                    (1) フレアクラス表現
                    """
                    # フレアクラス表現
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?(フレア|クラス)?', line):
                        line = clean_class(line)
                    
                    str_summary += line + '\n'

                elif re.search(r'[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?頃?に、?', line):
                    """
                    (1) 活動領域XXXX(NXXNXX) -> 活動領域
                    (2) 日付表記
                    (3) フレアクラス表現
                    """
                    # 活動領域
                    if re.search(r'活動領域[０-９]{4}(（([Ａ-Ｚ][０-９]{2}){1,3}）)?', line):
                        line = re.sub(r'活動領域[０-９]{4}(（([Ａ-Ｚ][０-９]{2}){1,3}）)?', r"活動領域", line)
                    
                    # 日付表現
                    line = clean_date_description(day, line)
                    # フレアクラス表現
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?(フレア|クラス)?', line):
                        line = clean_class(line)
                    
                    str_summary += line + '\n'
                
                elif re.search(r'活動領域[０-９]{4}', line):
                    """
                    (1) 活動領域XXXX(、XXXX){0,4}など -> 複数の活動領域
                    (2) 活動領域XXXX -> 活動領域XXXX
                    (3) フレアクラス表現
                    """
                    if re.search(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}など', line):
                        line = re.sub(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}など', r'複数の活動領域', line)
                    if re.search(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}', line):
                        line = re.sub(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}', r'活動領域', line)
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?(フレア|クラス)?', line):
                        line = clean_class(line)
                    if re.search(r'クラスのＬＤＥフレア（.*）', line):
                        line = re.sub(r'クラスのＬＤＥフレア（.*）', r'クラス', line)
                    if re.search(r'[０-９]{1,2}から[０-９]{1,2}に', line):
                        line = re.sub(r'[０-９]{1,2}から[０-９]{1,2}に', r'', line)
                    str_summary += line + '\n'
                
                else:
                    str_summary += line + "\n"

        elif index == 2:  # ---で区切られた段落のうちの太陽活動の詳細
            lines = text.split('\n')
            skip_flag = False
            for line in lines:
                str_detail_sun += line + '\n'
                # if re.match(r'<font',line):
                #     str_etc += line + "\n"
                #     skip_flag = False
                # elif re.match(r'^■',line):
                #     str_etc += line + "\n"
                #     skip_flag = True
                # elif re.match(r'^（参考データ）',line):
                #     str_etc += line + "\n"
                #     skip_flag = True
                # elif len(line) == 0:
                #     str_etc += line + "\n"
                # elif re.match(r'.*日間、地磁気.*',line):
                #     pass
                # elif re.match(r'^[０-９]{1,2}日[０-９]{1,2}時(（ＵＴ）)?現在、', line):
                #     str_detail_sun += re.sub(r'^[０-９]{1,2}日[０-９]{1,2}時(（ＵＴ）)?現在、', r'', line) + "\n"
                # elif re.match(r'^活動領域[０-９]{1,4}(、[０-９]{1,4})*', line):
                #     str_detail_sun += re.sub(r'^活動領域[０-９]{1,4}(、[０-９]{1,4})*', r'活動領域', line) + "\n"
                # else:
                #     if skip_flag:
                #         str_etc += line + "\n"
                #     else:
                #         str_detail_sun += line + "\n"

        elif index == 3:  # ---で区切られた段落のうちの地磁気活動の詳細
            lines = text.split('\n')
            skip_flag = False
            for line in lines:
                if re.match(r'<font',line):
                    str_etc += line + "\n"
                    skip_flag = False
                elif re.match(r'^■',line):
                    str_etc += line + "\n"
                    skip_flag = True
                elif re.match(r'^（参考データ）',line):
                    str_etc += line + "\n"
                    skip_flag = True
                elif len(line) == 0:
                    str_etc += line + "\n"
                else:
                    if skip_flag:
                        str_etc += line + "\n"
                    else:
                        str_detail_geomagnetism += line + "\n"
            break

        else:
            str_etc += text
        index = index+1

    # ファイル出力
    label_path = input_path.split('/')[-1]
    with open(output_path + label_path, "w") as output_file:
        output_file.write(str_summary)

    # 太陽詳細部分の保存
    # label_path = input_path.split('/')[-1]
    # output_path = '/home/initial/Flareformer/data_forecast/flare_detail/'
    # with open(output_path+label_path, 'w') as output_file:
    #     output_file.write(str_detail_sun)
    

def divide_templete(txt_file, output_path):
    """
    Args:
        txt_file : /home/initial/Flareformer/data_forecast/defn_labels/XXX.txt
        output_path;  /home/initial/Flareformer/data_forecast/defn_labels/
    """
    with open(txt_file, 'r') as f:
        text = f.read()
    
    template = ''
    predict = ''
    contexts = text.split('\n')
    for idx in range(len(contexts)):
        line = contexts[idx]
        if idx==0 or idx==1:
            template += line
        elif re.search(r'.*な状態が予想されます。$', line):
            template += line
        elif re.search(r'.*(静穏|活発)?でしょう', line):
            template += line
        else:
            predict += line
    
    txt_name = txt_file.split('/')[-1].split('.')[0]
    template_path = output_path + txt_name + "_template.txt"
    predict_path = output_path + txt_name + "_predict.txt"
    
    with open(template_path, 'w') as f:
        f.write(template)
    with open(predict_path, 'w') as f:
        f.write(predict)


if __name__ == "__main__":
    parse_forecast()
