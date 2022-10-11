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

def delete_month_time(line):
    """
    Args:
        line (str): parser対象の文字列
    
    (1) (XX月)?XX日(XX時)?(XX分)?(UT)?頃?に -> XX日XX時頃?に 
    と変形して返す関数
    """
    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
        line = re.sub(r'([０-９]{1,2}月)?', r'', line)
        line = re.sub(r'([０-９]{1,2}分)?(（ＵＴ）)?', r'', line)
    
    return line


def parse_forecast(input_path, output_path):
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
        if index == 1:  # ---で区切られた段落のうちの概要
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
                elif re.match(r'^地磁気', line):
                    pass
                elif re.match(r'.*日間、地磁気.*', line):
                    pass
                elif re.search(r'地磁気..', line):
                    pass
                elif re.search(r'臨時警報', line):
                    pass
                elif re.search(r"コロナホール", line):
                    pass
                elif re.match(r".*高エネルギー電子", line):
                    pass
                elif re.search(r".始型地?磁気嵐", line):
                    """
                    (1) (XX月)?XX日(XX時)?(XX分)?に -> XX日XX時に
                    (2) (XX月)?XX日(XX時)?(XX分)?頃に -> XX日XX時頃に
                    """
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?', line):
                        line = delete_month_time(line)
                    str_summary += line + "\n"
                elif re.search(r"プロトン..", line):
                    pass
                elif re.search(r'継続中', line):
                    pass
                elif re.search(r"太陽風速度は?", line):
                    """
                    (1) (XX月)?XX日(XX時)?(XX分)?(UT)?頃?(に|から) -> XX日(XX時)?頃?(に|から)
                    (2) (XX月)?XX日(XX時)?(XX分)?(UT)?現在、 -> XX日XX時現在、
                    
                    (2) (低、高)速なXXX km/sからXXX km/sへ -> (低、高)速な状態から に置換
                    (3) (低、高)速なXXX km/s(前後|付近)?(へ|で)? -> (低、高)速な状態へ に置換
                    (4) 通常速度 のXXXkm/s[前後] -> 通常速度の[前後]
                    (5) [太陽風速度は]XXXkm/s(前後|付近)(へ|を|の)? -> [太陽風速度は]
                    
                    (6) -nT(前後|付近)の -> 削除
                    (7) XXnT(前後|付近)(へ|まで|を) -> 状態 に置換
                    
                    (8) CME(コロナ質量放出) -> コロナ質量放出
                    (9) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
                    (10) 発生した、 -> 削除 (先頭)
                    """

                    # XX月XX日関連
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?頃?(に|から)', line):
                        line = delete_month_time(line)
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?現在、?', line):
                        line = delete_month_time(line)

                    # 速度関係
                    if re.search(r'(低|高)速な[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後)?から[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後)?へ', line):
                        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後)?から[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後)?へ', r'状態から', line)
                    if re.search(r'(低|高)速な[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近)?(へ|で)?', line):
                        line = re.sub(r'速な[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近)?', r'速な状態', line)
                    if re.search(r'通常速度の[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ]', line):
                        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ]', r'', line)
                    if re.search(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近)?(へ|を|の)', line):
                        line = re.sub(r'[０-９]{1,3}[ａ-ｚ]{1,2}／[ａ-ｚ](前後|付近)?(へ|を|の)', r'', line)

                    # nT関連
                    if re.search(r'−?[０-９]{1,2}ｎＴ(前後|付近)?の', line):
                        line = re.sub(r'−?[０-９]{1,2}ｎＴ(前後|付近)?の', r'', line)
                    if re.search(r'[０-９]{1,2}ｎＴ(前後|付近)?(へ|まで|を)', line):
                        line = re.sub(r'[０-９]{1,2}ｎＴ(前後|付近)?', r'状態', line)
                                        
                    # その他
                    if re.search(r'ＣＭＥ(（コロナ質量放出）)?', line):
                        line = re.sub(r'ＣＭＥ(（コロナ質量放出）)?', r'コロナ質量放出', line)
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?', line):
                        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
                    if re.match(r'発生した、', line):
                        line = re.sub(r'発生した、', r'', line)
                    str_summary += line + '\n'

                elif re.search(r'太陽風の?磁場', line):
                    """
                    (1) -XXnT(前後|付近)の -> 状態 に置換
                    (2) XXnT(前後|付近)? -> 状態 に置換
                    (3) (XX月)?XX日(XX時)?(XX分)?(UT)?現在、 -> XX日XX時現在、 に置換
                    (4) (XX月)?XX日(XX時)?(XX分)?(UT)?頃?(に|から)、? -> XX日XX時頃(に|から)、?に置換
                    (5) CME(コロナ質量放出) -> コロナ質量放出

                    """
                    # nT関連
                    if re.search(r'−?[０-９]{1,2}ｎＴ(前後|付近)?の', line):
                        line = re.sub(r'−?[０-９]{1,2}ｎＴ(前後|付近)?の', r'', line)
                    if re.search(r'[０-９]{1,2}ｎＴ(前後|付近)?', line):
                        line = re.sub(r'[０-９]{1,2}ｎＴ(前後|付近)?', r'状態', line)
                    
                    # XX月XX日関連
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?現在、', line):
                        line = delete_month_time(line)
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?頃?(に|から)、?', line):
                        line = delete_month_time(line)
                    
                    # CME関連
                    if re.search(r'ＣＭＥ(（コロナ質量放出）)?', line):
                        line = re.sub(r'ＣＭＥ(（コロナ質量放出）)?', r'コロナ質量放出', line)
                    str_summary += line + "\n"

                elif re.search(r'ＣＭＥ', line):
                    """
                    (1) CME(コロナ質量放出) -> コロナ質量放出
                    (2) XX日XX時(UR)頃とXX日XX時(UT)頃に発生したコロナ質量放出 -> XX日XX時頃とXX日XX時頃に発生した に置換
                    (3) (XX月)?XX日からXX日頃?に(、|かけて) -> XX日からXX日頃?に(、|かけて) 日間
                    (4) (XX月)?XX日(XX時)?(XX分)?頃?に(かけて)? -> XX日(XX時)?頃?に(かけて)? に置換
                    (5) (XX月)?XX日(XX時)?(XX分)?頃?の -> XX日(XX分)?頃?の に置換
                    (6) 発生したCMEと発生したCME -> 複数のCME
                    (7) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
                    """
                    line = re.sub(r'ＣＭＥ(（コロナ質量放出）)?', r'コロナ質量放出', line)

                    if re.search(r'[０-９]{1,2}日[０-９]{1,2}時(（ＵＴ）)?頃と[０-９]{1,2}日[０-９]{1,2}時(（ＵＴ）)?頃に発生した', line):
                        line = re.sub(r'(（ＵＴ）)?', r'', line)
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日から[０-９]{1,2}日頃?に(、|かけて)?', line):
                        line = re.sub(r'([０-９]{1,2}月)?', r'', line)
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?頃?に(かけて)?、?', line):
                        line = delete_month_time(line)
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?頃?の', line):
                        line = delete_month_time(line)
                    if re.search(r'発生したＣＭＥと、発生したＣＭＥ', line):
                        line = re.sub(r'発生したＣＭＥと、発生したＣＭＥ', r'複数のＣＭＥ', line)
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?', line):
                        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
                    str_summary += line + "\n"

                elif re.search(r'(この)|(最大の)フレアは、', line):
                    """
                    (1) 活動領域XXXX -> 活動領域
                    (2) XX月XX日XX分(UT)に -> XX日に に置換
                    (3) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
                    (4) 野辺山電波...によると、 -> 削除
                    """
                    if re.search(r'活動領域[０-９]{4}', line):
                        line = re.sub(r'活動領域[０-９]{4}', r'活動領域', line)
                    if re.search(r'([０-９]{1,2}月)?[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?に', line):
                        line = delete_month_time(line)
                    if re.search(r'[０-９](．[０-９])?(／[０-９][Ａ-Ｚ])?', line):
                        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
                    if re.search(r'野辺山電波.*よると、', line):
                        line = re.sub('野辺山電波.*よると、', r'', line)
                    str_summary += line + '\n'
                
                elif re.match(r'太陽面上?の?.*端で', line):
                    str_summary += line + '\n'
                
                elif re.match(r'太陽面では?', line):
                    """
                    (1) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
                    (2) (X|M|B|C)Nクラス -> (X|M|B|C)クラス
                    """
                    if re.search(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', line):
                        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
                    if re.search(r'[０-９]クラス', line):
                        line = re.sub(r'[０-９]クラス', r'クラス', line)
                    str_summary += line + '\n'

                elif re.search(r'[０-９]{1,2}日([０-９]{1,2}時)?([０-９]{1,2}分)?(（ＵＴ）)?頃?に、?', line):
                    """
                    (1) XX日(XX時)?(XX分)?(UT)頃?に、? -> XX日(XX分)頃?に に置換
                    (2) 活動領域XXXX(NXXNXX) -> 活動領域
                    (3) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
                    """
                    line = delete_month_time(line)
                    if re.search(r'活動領域[０-９]{4}(（([Ａ-Ｚ][０-９]{2}){1,3}）)?', line):
                        line = re.sub(r'活動領域[０-９]{4}(（([Ａ-Ｚ][０-９]{2}){1,3}）)?', r"活動領域", line)
                    if re.search(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', line):
                        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
                    str_summary += line + '\n'
                
                elif re.search(r'活動領域[０-９]{4}', line):
                    """
                    (1) 活動領域XXXX(、XXXX){0,4}など -> 複数の活動領域
                    (2) 活動領域XXXX -> 活動領域XXXX
                    (3) (X|M|B|C)N.Nフレア -> (X|M|B|C)クラスフレア
                    (4) (X|M|B|C)クラスのLDEフレア(.*) -> (X|M|B|C)クラス
                    """
                    if re.search(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}など', line):
                        line = re.sub(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}など', r'複数の活動領域', line)
                    if re.search(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}', line):
                        line = re.sub(r'活動領域[０-９]{4}(、[０-９]{4}){0,4}', r'活動領域', line)
                    if re.search(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', line):
                        line = re.sub(r'[０-９]．[０-９](／[０-９][Ａ-Ｚ])?', r'クラス', line)
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
                elif re.match(r'.*日間、地磁気.*',line):
                    pass
                elif re.match(r'^[０-９]{1,2}日[０-９]{1,2}時(（ＵＴ）)?現在、', line):
                    str_detail_sun += re.sub(r'^[０-９]{1,2}日[０-９]{1,2}時(（ＵＴ）)?現在、', r'', line) + "\n"
                elif re.match(r'^活動領域[０-９]{1,4}(、[０-９]{1,4})*', line):
                    str_detail_sun += re.sub(r'^活動領域[０-９]{1,4}(、[０-９]{1,4})*', r'活動領域', line) + "\n"
                else:
                    if skip_flag:
                        str_etc += line + "\n"
                    else:
                        str_detail_sun += line + "\n"

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
