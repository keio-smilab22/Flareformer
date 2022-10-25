import glob
import pandas as pd
import cv2
import os


def main(csv_path, data_path, video_path, span=24):

    # read csv by pandas
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    filename = df['filename'].values
    for i in range(len(filename)-span):
        img_array = []
        for j in range(span):
            # filename example: hmi_m_45s_2011_01_01_00_01_30_tai_magnetogram.fits.png
            year = filename[i+j].split('_')[3]
            month = filename[i+j].split('_')[4]
            # print(f"filename: {filename[i+j]}")
            file_path = os.path.join(data_path, year, month, filename[i+j])
            img = cv2.imread(file_path)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        name = os.path.join(video_path, year, month, filename[i].split('.')[0]+'.mp4')
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m','p','4', 'v'), 12, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        print('video {} is done'.format(name))

if __name__ == '__main__':
    span = 24
    csv_path = 'data/noaa/magnetogram_logxmax1h_all_years.csv'
    data_path = 'data/noaa/magnetogram'
    video_path = f'data/noaa/magnetogram_video_{span}'
    main(csv_path, data_path, video_path)
