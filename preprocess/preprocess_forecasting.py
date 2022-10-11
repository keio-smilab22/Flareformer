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


def main(args: argparse.Namespace):
    txt_list = sorted(glob(args.input_path+"/*.txt"))
    for txt in tqdm(txt_list, desc="parse forecast: "):
        parse_forecast(input_path=txt, output_path=args.output_path)

    label_list = sorted(glob(args.output_path+"*.txt"))
    for idx, txt in enumerate(tqdm(label_list, desc='divide templete')):
        divide_templete(txt, args.output_path)
    
    print(args.output_path)
    print()
    print('##### Save label by year #####')
    make_label_by_year(args.output_path)
    print('##### Save all label to json #####')
    all_label_to_json(args.output_path)



if __name__ == "__main__":
    main(parse_args())