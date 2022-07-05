from pickle import FALSE
import matplotlib.pyplot as plt

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import astropy.units as u
import numpy as np
import os
from tqdm import tqdm
import cv2

def create_magentogram(path_d, path_save, month=1):
    # result = Fido.search(a.Time('2011/01/01 00:00:00', '2011/01/01 00:30:00'),
    #                  a.Instrument.hmi, a.Physobs.los_magnetic_field, a.Sample(10*u.minute))
    month_span ={1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    start_date = f'2011/{month:02d}/01'
    end_date = f'2011/{month:02d}/{month_span[month]:02d}'
    result = Fido.search(a.Time(start_date, end_date),
                     a.Instrument.hmi, a.Physobs.los_magnetic_field, a.Sample(15*u.minute))
    print(result)
    downloaded_file = Fido.fetch(result, max_conn=100, overwrite=False)
    while True:
        if len(downloaded_file.errors) == 0:
            break

        downloaded_file = Fido.fetch(downloaded_file, max_conn=100, overwrite=False)
    
    
    # for i in range(100):
    #     if len(downloaded_file.errors) == 0:
    #         break

    #     downloaded_file = Fido.fetch(downloaded_file, max_conn=100, overwrite=False)

    for file in tqdm(downloaded_file, desc='Creating magentogram', total=len(downloaded_file)):
        try :
            hmi_map = sunpy.map.Map(file)
            # plt.figure()
            # hmi_map.plot()
            # plt.show()

            hmi_rotated = hmi_map.rotate(order=3)
            # plt.figure()
            # save with the same name as the downloaded file and no axis
            figsize_px = np.array([1024, 1024])
            dpi = 300
            figsize_inch = figsize_px / dpi
            figure = plt.figure(frameon=False, figsize=figsize_inch, dpi=dpi)
            ax = plt.axes([0, 0, 1, 1])
            # Disable the axis
            ax.set_axis_off()

            # Plot the map. Since are not interested in the exact map coordinates, we can
            # simply use :meth:`~matplotlib.Axes.imshow`.
            # norm = hmi_rotated.plot_settings['norm']
            # norm.vmin, norm.vmax = np.percentile(hmi_rotated.data, [0, 100])
            # print(hmi_rotated.plot_settings['cmap'])
            ax.imshow(hmi_rotated.data,
                    cmap=hmi_rotated.plot_settings['cmap'],
                    origin="lower",
                    norm=hmi_rotated.plot_settings['norm'])
            print(file)
            filename = file + '.png'
            # ax.imshow(hmi_rotated.data)
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            plt.savefig(os.path.join(path_save, os.path.basename(filename)), bbox_inches='tight', pad_inches=0)
            plt.cla()
            # plt.clf()
            plt.close(figure)
        except Exception as e:
            print(e)
            continue
        # plt.show()

def test_magnetogram(path_d):
    file_path = os.path.join(path_d, 'hmi_m_45s_2011_01_01_00_41_15_tai_magnetogram.png')
    # load the image
    img = cv2.imread(file_path, -1)
    img_p = plt.imread(file_path)
    print(img_p.shape)
    print(img.shape)
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    print(gray[500, 500])
    print(gray.shape)
    print(gray[0, 0])
    # show the image
    fig, ax = plt.subplots()
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    path_d = 'data/noaa/'
    path_save = 'data/noaa/magnetogram/2011/'
    for i in range(9, 13):
        create_magentogram(path_d, path_save, i)
        print(f'Month {i} done')
    # test_magnetogram(path_save)