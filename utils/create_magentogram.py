import datetime
import matplotlib.pyplot as plt

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import astropy.units as u
import numpy as np
import os
from tqdm import tqdm
import cv2

def create_magentogram(path_save, year=2011, month=1):
    # result = Fido.search(a.Time('2011/01/01 00:00:00', '2011/01/01 00:30:00'),
    #                  a.Instrument.hmi, a.Physobs.los_magnetic_field, a.Sample(10*u.minute))
    
    # if year is the leap year, then 
    if year % 4 == 0:
        month_span = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    else:
        month_span = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

    for day in tqdm(range(1, month_span[month]+1), desc='day'):
        start_date = f'{year}/{month:02d}/{day:02d}'
        # next datetime
        start_date = datetime.datetime.strptime(start_date, '%Y/%m/%d')
        end_date = start_date + datetime.timedelta(days=1)
        end_date = end_date.strftime('%Y/%m/%d')

        # end_date = f'{year}/{month+1:02d}/{1:02d}'

        result = Fido.search(a.Time(start_date, end_date),
                        a.Instrument.hmi, a.Physobs.los_magnetic_field, a.Sample(5*u.minute))
        print(result)
        downloaded_file = Fido.fetch(result, max_conn=100, overwrite=False)
        while True:
            if len(downloaded_file.errors) == 0:
                break

            downloaded_file = Fido.fetch(downloaded_file, max_conn=100, overwrite=False)
        
        fig, ax = plt.subplots(frameon=False)
        
        for file in tqdm(downloaded_file, desc='Creating magentogram', total=len(downloaded_file)):
            try :
                
                # save with the same name as the downloaded file and no axis
                save_dir = os.path.join(path_save, f'{month:02d}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                filename = file + '.png'
                # if file is already exist, skip it and if not, save it
                if os.path.exists(os.path.join(save_dir, os.path.basename(filename))):
                    print(f'{filename} already exist')
                    continue
                else:
                    # Plot the map. Since are not interested in the exact map coordinates, we can
                    # simply use :meth:`~matplotlib.Axes.imshow`.
                    # norm = hmi_rotated.plot_settings['norm']
                    # norm.vmin, norm.vmax = np.percentile(hmi_rotated.data, [0, 100])
                    # print(hmi_rotated.plot_settings['cmap'])W
                    hmi_map = sunpy.map.Map(file)
                    hmi_rotated = hmi_map.rotate(order=3)

                    # ax = plt.axes([0, 0, 1, 1])
                    # Disable the axis
                    ax.set_axis_off()
                    ax.imshow(hmi_rotated.data,
                            cmap=hmi_rotated.plot_settings['cmap'],
                            origin="lower",
                            norm=hmi_rotated.plot_settings['norm'])
                    
                    # ax.imshow(hmi_rotated.data)
                    
                    # plt.savefig(os.path.join(save_dir, os.path.basename(filename)), bbox_inches='tight', pad_inches=0)
                    
                    plt.savefig(os.path.join(save_dir, os.path.basename(filename)), bbox_inches='tight', pad_inches=0)
                    print(f'{filename} saved')
                    plt.cla()
                    # plt.close(figure)
                
            except Exception as e:
                print(e)
                plt.cla()
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
    year = 2013
    path_save = f'data/noaa/magnetogram/{year}_5m/'
    for i in range(2,5):
        create_magentogram(path_save=path_save, year=year, month=i)
        print(f'Month {i} done')
    # test_magnetogram(path_save)