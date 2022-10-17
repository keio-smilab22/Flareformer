"""
太陽フレア予報文の前処理を行う
"""
import argparse
import csv
import re
import json
from glob import glob

from tqdm import tqdm

from parse import parse_forecast, divide_templete

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', '-i', type=str, default='/home/initial/Flareformer/data_forecast/defn_txt_2011-17')
    parser.add_argument('--output_path', '-o', type=str, default='/home/initial/Flareformer/data_forecast/defn_labels/')
    parser.add_argument('--save_entity', action="store_true")

    return parser.parse_args()

def extract_label(lines: str):
    """
    <AA>太陽活動は<BB>でした。<CC>太陽活動は<DD>な状態が予想されます
    テキストから<AA> <BB>に該当する部分を抽出
    (なし, B, C, M, X) -> (0, 1, 2, 3, 4)
    """
    AA, BB = [0,0,0,0,0], [0,0]
    line = lines.split('。')[0]
    if re.search(r'Ｘ', line):
        AA[4] = 1
    elif re.search(r'Ｍ', line):
        AA[3] = 1
    elif re.search(r'Ｃ', line):
        AA[2] = 1
    elif re.search(r'Ｂ', line):
        AA[1] = 1
    else:
        AA[0] = 1
    
    if re.search(r'活発', line):
        BB[1] = 1
    else:
        BB[0] = 1
    
    assert 1 in (AA and BB)
    return AA, BB

def all_label_to_json(data_path: str):
    template_list = sorted(glob(data_path+'*_template.txt'))
    save_path = data_path + 'template_labels.jsonl'
    for idx, template in enumerate(template_list):
        file_name = template.split("/")[-1].replace("_template", '')
        
        with open(template, 'r') as f:
            lines = f.read()
        AA, BB = extract_label(lines)

        label = {
            'file': file_name,
            'class_': AA,
            'active': BB,
        }

        with open(save_path, 'a') as f:
            json.dump(label, f)
            f.write('\n')

def make_label_by_year(data_path: str):
    years = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
    for year in years:
        template_list = sorted(glob(data_path+f'd{year}*_template.txt'))

        for idx, template in enumerate(template_list):
            with open(template, 'r') as f:
                lines = f.read()
            AA, BB = extract_label(lines)

            # save label
            save_path = data_path + f'data_{year}_forecast_label.csv'
            with open(save_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(AA)

            # save active
            save_path = data_path + f'data_{year}_forecast_active.csv'
            with open(save_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(BB)


def to_3part(
    files_path: str = "",
    output_path: str = "",
):
    """
        予報をを
            (1) 太陽動動に関する記述 (太陽活動 or 活動領域 or フレアを含む文)
            (2) 地磁気動動に関する記述 (地磁気を含む文)
            (3) それ以外の文
        に分割する
    """

    texts = sorted(glob(files_path+'*_web.txt'))
    for idx, text in enumerate(texts):
        labels_flare = ""
        labels_mag = ""
        labels_other = ""
        file_name = text.split("/")[-1].replace("_web.txt", "")

        with open(text, "r") as f:
            lines = f.read()
        
        output_flare = output_path + file_name + "_flare.txt"
        output_mag = output_path + file_name + "_mag.txt"
        output_other = output_path + file_name + "_other.txt"

        lines = lines.split()
        for line in lines:
            if re.search(r"太陽活動", line):
                labels_flare += line + "\n"
            elif re.search(r'活動領域', line):
                labels_flare += line + "\n"
            elif re.search(r'このフレアは', line):
                labels_flare += line + "\n"
            elif re.search(r'(Ｍ|Ｘ|Ｃ|Ｂ)フレアが発生', line):
                labels_flare += line + "\n"
            elif re.search(r"地磁気", line):
                labels_mag += line + "\n"
            else:
                labels_other += line + "\n"
        
        with open(output_flare, "a") as f:
            f.write(labels_flare)
        with open(output_mag, "a") as f:
            f.write(labels_mag)
        with open(output_other, "a") as f:
            f.write(labels_other)


def count_other(
    files_path: str,
):
    count = 0
    texts = sorted(glob(files_path+'*_other.txt'))
    for idx, text in enumerate(texts):
        file_name = text.split("/")[-1].replace("_web.txt", "")
        with open(text, "r") as f:
            lines = f.read()
        
        if len(lines) > 0:
            count += 1
    
    print(len(texts))
    return count

def count_stat_flare(files_path: str):
    stat = {
        "Xクラス": 0,   "Mクラス": 0,       "Cクラス": 0,
        "Bクラス": 0,   "活発": 0,          "静穏": 0,
        "黒点": 0,      "予想": 0,          "東西南北": 0,
        "裏側": 0,      "磁場構造": 0,       "面積": 0,
        "太陽風": 0,    "磁場強度": 0,       "コロナホール": 0,
        "磁場の成分": 0, "コロナ質量放出": 0,  "子午線": 0,
        "衝撃波": 0,    "プロトン現象": 0,    "急始型": 0,
        "緩始型": 0,    "磁気嵐": 0,
    }
    num = 0
    texts = sorted(glob(files_path+"d*_flare.txt"))
    for idx, text in enumerate(texts):
        with open(text, "r") as f:
            lines = f.read()

        if len(lines) > 0: num += 1
        lines = lines.split()
        for line in lines:
            if re.search(r"Ｂクラス", line):
                stat["Bクラス"] += 1
            if re.search(r"Ｃクラス", line):
                stat["Cクラス"] += 1
            if re.search(r"Ｍクラス", line):
                stat["Mクラス"] += 1
            if re.search(r"Ｘクラス", line):
                stat["Xクラス"] += 1
            if re.search(r"活発", line):
                stat["活発"] += 1
            if re.search(r"静穏", line):
                stat["静穏"] += 1
            if re.search(r"黒点", line):
                stat["黒点"] += 1
            if re.search(r"予想", line):
                stat["予想"] += 1
            if re.search(r"東|西|南|北", line):
                stat["東西南北"] += 1
            if re.search(r"裏側", line):
                stat["裏側"] += 1
            if re.search(r"磁場構造", line):
                stat["磁場構造"] += 1
            if re.search(r"面積", line):
                stat["面積"] += 1
            if re.search(r'太陽風', line):
                stat['太陽風'] += 1
            if re.search(r'磁場強度', line):
                stat['磁場強度'] += 1
            if re.search(r'コロナホール',line):
                stat['コロナホール'] += 1
            if re.search(r"磁場の..成分", line):
                stat["磁場の成分"] += 1
            if re.search(r"コロナ質量放出", line):
                stat["コロナ質量放出"] += 1
            if re.search(r"子午線", line):
                stat["子午線"] += 1
            if re.search(r"衝撃波", line):
                stat["衝撃波"] += 1
            if re.search(r"プロトン現象", line):
                stat["プロトン現象"] += 1
            if re.search(r"急始型", line):
                stat["急始型"] += 1
            if re.search(r"緩始型", line):
                stat["緩始型"] += 1
            if re.search(r"磁気嵐", line):
                stat["磁気嵐"] += 1
        
    return (stat, num)


def count_stat_mag(
    files_path: str,
    is_save: bool = True,
    ):
    """
        Error文
            今後とも静穏でしょう。
            本日Ｍクラスフレアが発生しました。
            昨日発生した急始型の影響は、現在も継続中です。
    """
    stat = {
        "太陽風": 0,    "磁場強度": 0,      "コロナホール": 0,
        "東西南北": 0,  "磁場の成分": 0,    "コロナ質量放出": 0,
        "子午線": 0,    "衝撃波": 0,        "プロトン現象": 0,
        "急始型": 0,    "緩始型": 0,        "磁気嵐": 0,
    }
    # 保存用
    str_velocity, str_mag, str_colo = "", "", ""
    str_touzai, str_seibun, str_cme = "", "", ""
    str_syougeki, str_proton, str_storm = "", "", ""
    num = 0
    texts = sorted(glob(files_path+"d*_mag.txt"))
    for idx, text in enumerate(texts):
        with open(text, "r") as f:
            lines = f.read()
        
        if len(lines) > 0: num+=1
        lines = lines.split()

        for line in lines:
            if re.search(r"太陽風", line):
                stat["太陽風"] += 1
                str_velocity += line + '\n'
            if re.search(r"磁場強度", line):
                stat["磁場強度"] += 1
                str_mag += line + '\n'
            if re.search(r"コロナホール", line):
                stat["コロナホール"] += 1
                str_colo += line + '\n'
            if re.search(r"東|西|南|北", line):
                stat["東西南北"] += 1
                str_touzai += line + '\n'
            if re.search(r"磁場の..成分", line):
                stat["磁場の成分"] += 1
                str_seibun += line + '\n'
            if re.search(r"コロナ質量放出", line):
                stat["コロナ質量放出"] += 1
                str_cme += line + '\n'
            if re.search(r"子午線", line):
                stat["子午線"] += 1
            if re.search(r"衝撃波", line):
                stat["衝撃波"] += 1
                str_syougeki += line + '\n'
            if re.search(r"プロトン現象", line):
                stat["プロトン現象"] += 1
                str_proton += line + '\n'
            if re.search(r"急始型", line):
                stat["急始型"] += 1
            if re.search(r"緩始型", line):
                stat["緩始型"] += 1
            if re.search(r"磁気嵐", line):
                stat["磁気嵐"] += 1
                str_storm += line + '\n'
    # other内用として保存
    if is_save:
        save_path = files_path + "entity/"
        with open(save_path+"vel_mag.txt", 'w') as f:
            f.write(str_velocity)
        with open(save_path+"mag_mag.txt", 'w') as f:
            f.write(str_mag)
        with open(save_path+"corhole_mag.txt", 'w') as f:
            f.write(str_colo)
        with open(save_path+"direction_mag.txt", 'w') as f:
            f.write(str_touzai)
        with open(save_path+'magcomp_mag.txt', 'w') as f:
            f.write(str_seibun)
        with open(save_path+'cme_mag.txt', 'w') as f:
            f.write(str_cme)
        with open(save_path+'shockwave_mag.txt', 'w') as f:
            f.write(str_syougeki)
        with open(save_path+'proton_mag.txt', 'w') as f:
            f.write(str_proton)
        with open(save_path+'storm_mag.txt', 'w') as f:
            f.write(str_storm)
    
    return (stat, num)


def count_stat_other(
    files_path: str,
    is_save: bool=True,
    ):
    """
        Error文
            今後とも静穏でしょう。
            本日Ｍクラスフレアが発生しました。
            昨日発生した急始型の影響は、現在も継続中です。

    """
    stat = {
        "太陽風": 0,    "磁場強度": 0,  "コロナホール": 0,
        "東西南北": 0,  "磁場の成分": 0, "コロナ質量放出": 0,
        "子午線": 0,    "衝撃波": 0,    "プロトン現象": 0,
        "急始型": 0,    "緩始型": 0,    "磁気嵐": 0,
    }
    # 保存用
    str_velocity, str_mag, str_colo = "", "", ""
    str_touzai, str_seibun, str_cme = "", "", ""
    str_syougeki, str_proton, str_storm = "", "", ""

    num = 0
    texts = sorted(glob(files_path+"d*_other.txt"))
    for idx, text in enumerate(texts):
        with open(text, "r") as f:
            lines = f.read()
        
        # 何もないものは飛ばす
        if len(lines) == 0: continue
        
        num += 1
        lines = lines.split()

        for line in lines:
            if re.search(r"太陽風", line):
                str_velocity += line + '\n'
            if re.search(r"磁場強度", line):
                stat["磁場強度"] += 1
                str_mag += line + '\n'
            if re.search(r"コロナホール", line):
                stat["コロナホール"] += 1
                str_colo += line + '\n'
            if re.search(r"東|西|南|北", line):
                stat["東西南北"] += 1
                str_touzai += line + '\n'
            if re.search(r"磁場の..成分", line):
                stat["磁場の成分"] += 1
                str_seibun += line + '\n'
            if re.search(r"コロナ質量放出", line):
                stat["コロナ質量放出"] += 1
                str_cme += line + '\n'
            if re.search(r"子午線", line):
                stat["子午線"] += 1
            if re.search(r"衝撃波", line):
                stat["衝撃波"] += 1
                str_syougeki += line + '\n'
            if re.search(r"プロトン現象", line):
                stat["プロトン現象"] += 1
                str_proton += line + '\n'
            if re.search(r"急始型", line):
                stat["急始型"] += 1
            if re.search(r"緩始型", line):
                stat["緩始型"] += 1
            if re.search(r"磁気嵐", line):
                stat["磁気嵐"] += 1
                str_storm += line + '\n'
        
    # other内用として保存
    if is_save:
        save_path = files_path + "entity/"
        with open(save_path+"vel_other.txt", 'w') as f:
            f.write(str_velocity)
        with open(save_path+"mag_other.txt", 'w') as f:
            f.write(str_mag)
        with open(save_path+"corhole_other.txt", 'w') as f:
            f.write(str_colo)
        with open(save_path+"direction_other.txt", 'w') as f:
            f.write(str_touzai)
        with open(save_path+'magcomp_other.txt', 'w') as f:
            f.write(str_seibun)
        with open(save_path+'cme_other.txt', 'w') as f:
            f.write(str_cme)
        with open(save_path+'shockwave_other.txt', 'w') as f:
            f.write(str_syougeki)
        with open(save_path+'proton_other.txt', 'w') as f:
            f.write(str_proton)
        with open(save_path+'storm_other.txt', 'w') as f:
            f.write(str_storm)
            
    return (stat, num)

def main(args: argparse.Namespace):
    # print(f'input_path >>> {args.input_path}')
    # txt_list = sorted(glob(args.input_path+"/*.txt"))
    # for txt in tqdm(txt_list, desc="parse forecast: "):
    #     parse_forecast(input_path=txt, output_path=args.output_path)
    # print(f'output_path >>> {args.output_path}')

    # # 太陽活動 / 地磁気 / その他に分割したファイをを作成
    # print("input files path >>> ", args.output_path)
    # print("output files path >>> ", args.output_path)
    # print("Devide texts in 3 parts >>>", end="")
    # to_3part(args.output_path, args.output_path)
    # print("ok")

    # その他のファイルにラベルを含む文の個数を求める
    # print('Count other part label nums >> ')
    # print('input files path >>> ', args.output_path)
    # count = count_other(args.output_path)
    # print(f'Other labels num >>> {count}')

    # 表現の数を求める
    print('input files path >>> ', args.output_path)
    print("Count statistice of flare part >> ")
    stat_flare, num_flare = count_stat_flare(args.output_path)
    print("Count statistice of flare part >> ")
    stat_mag, num_mag = count_stat_mag(args.output_path)
    print("Count statistice of other part >> ")
    state_other, num_other = count_stat_other(args.output_path)

    print("=======================================")
    print(num_flare)
    print(stat_flare)
    print("=======================================")
    print(num_mag)
    print(stat_mag)
    print("=======================================")
    print(num_other)
    print(state_other)
    print("=======================================")

    
    # print(f'output_path >>> {args.output_path}')
    # label_list = sorted(glob(args.output_path+"*.txt"))
    # for idx, txt in enumerate(tqdm(label_list, desc='divide templete')):
    #     divide_templete(txt, args.output_path)
    
    # print('##### Save label by year ##### >>> ', end='')
    # make_label_by_year(args.output_path)
    # print('ok')
    # print('##### Save all label to json ##### >>> ', end='')
    # all_label_to_json(args.output_path)
    # print('ok')

    # 3文目があるやつのカウント用
    # txt_list = sorted(glob(args.output_path+'*web.txt'))
    # num = 0
    # for file in txt_list:
    #     with open(file, 'r') as f:
    #         lines = f.read()
    #     line = lines.split()
    #     if len(line) > 2:
    #         txt = "".join(line)
    #         with open('/home/initial/Flareformer/data_forecast/defn_labels/test.txt', 'a') as ft:
    #             ft.write(txt)
    #             ft.write('\n')
    #         num += 1
    # print(f'{num=}')


if __name__ == "__main__":
    main(parse_args())