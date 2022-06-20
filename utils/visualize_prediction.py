import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def visualize_prediction(preds, trues, prev_seqs, base_dir):
    """
    Visualize the prediction and actual values.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    save_dir = os.path.join(base_dir, 'prediction_no_title')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # os.makedirs('result_images/FT_MAE_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_ffill_2016_0', exist_ok=True)
    basetime = datetime.datetime(2016, 1, 1, 0, 0, 0)


    min_mae = np.inf
    plt.figure()
    # same length array as prev_seqs but all nan
    nan_seqs = np.full(prev_seqs.shape, np.nan)

    preds = np.concatenate((prev_seqs, preds), axis=1)
    trues = np.concatenate((prev_seqs, trues), axis=1)

    for i in tqdm(range(preds.shape[0])):
        
        offset = - prev_seqs.shape[1] + 1
        x = np.arange(offset, preds.shape[1] + offset)
        mae = np.mean(np.abs(preds[i, prev_seqs.shape[1]:, -1] - trues[i, prev_seqs.shape[1]:, -1]))
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams["font.size"] = 20 #フォントの大きさ
        plt.rcParams["figure.subplot.bottom"] = 0.2 #下の余白


        plt.plot(x, trues[i, :, -1], label='(a)', color='blue', zorder=2)
        plt.plot(x, preds[i, :, -1], label='(b)', color='red', zorder=1)
        if mae < min_mae:
            min_mae = mae
            min_mae_idx = i
        # Time i hours after basetime
        current_time = basetime + datetime.timedelta(hours=i)
        # plt.title(f'MAE: {mae:.3f}, {current_time}')
        plt.ylabel('logXmax1h', fontsize=20)
        plt.xlabel('Time [h]', fontsize=20)
        plt.axvline(x=0, color='black', linestyle='--', zorder=0)
        plt.ylim(-3.0, 5.0)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{mae:.3f}_{i:04}_{current_time}.png')
        plt.cla()
        
    print(f'Min MAE: {min_mae} at {min_mae_idx}')

def visualize_prediction_sample(preds, trues, base_dir):
    """
    Visualize the prediction and actual values.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    faild_dir = os.path.join(base_dir, 'failed')
    if not os.path.exists(faild_dir):
        os.makedirs(faild_dir)

    # os.makedirs('result_images/FT_MAE_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_ffill_2016_0', exist_ok=True)
    
    min_mae = np.inf
    plt.figure()
    for i in tqdm(range(preds.shape[0])):
        
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.plot(trues[i, :, -1], label='(a)', color='blue')
        plt.plot(preds[i, :, -1], label='(b)', color='red')
        mae = np.mean(np.abs(preds[i, :, -1] - trues[i, :, -1]))
        if mae < min_mae:
            min_mae = mae
            min_mae_idx = i
        
        if mae > 0.5:
            plt.title(f'MAE: {mae:.3f}')
            plt.ylabel('logXmax1h')
            plt.xlabel('Time [h]')
            plt.ylim(-2.0, 6.0)
            plt.legend()
            plt.savefig(f'{faild_dir}/{mae:.3f}_{i:04}.png')

        plt.cla()
    print(f'Min MAE: {min_mae} at {min_mae_idx}')


def visualize_predictions_compare(preds1, trues1, preds2, trues2, base_dir):
    """
    Visualize the prediction and actual values.
    """
    compare_dir = os.path.join(base_dir, 'compare')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)

    plt.figure(figsize=(15,5))
    for i in tqdm(range(preds2.shape[0])):
        
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        
        plt.subplot(1, 2, 1)
        plt.plot(trues1[i, :, -1], label='(a)', color='blue')
        plt.plot(preds1[i, :, -1], label='(b)', color='red')
        mae1 = np.mean(np.abs(preds1[i, :, -1] - trues1[i, :, -1]))
        plt.title(f'MAE: {mae1:.3f}')
        plt.ylabel('logXmax1h')
        plt.xlabel('Time [h]')
        plt.ylim(-2.0, 6.0)
        plt.legend()
        

        plt.subplot(1, 2, 2)
        plt.plot(trues2[i, :, -1], label='(a)', color='blue')
        plt.plot(preds2[i, :, -1], label='(c)', color='red')
        mae2 = np.mean(np.abs(preds2[i, :, -1] - trues2[i, :, -1]))
        plt.title(f'MAE: {mae2:.3f}')
        plt.ylabel('logXmax1h')
        plt.xlabel('Time [h]')
        plt.ylim(-3, 5)
        plt.legend()
        if mae1 < mae2:
            plt.savefig(f'{compare_dir}/{mae1:.3f}_{i:04}.png')
        plt.clf()

def visualize_predictions(preds1, trues1, preds2, trues2):
    """
    Visualize the prediction and actual values.
    """
    

    
    for i in range(8000, preds.shape[0]):
        plt.figure(figsize=(15,5))
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.subplot(1, 2, 1)
        
        plt.plot(trues1[i, :, -1], label='(a)', color='blue')
        plt.plot(preds1[i, :, -1], label='(b)', color='red')
        mae = np.mean(np.abs(preds1[i, :, -1] - trues1[i, :, -1]))
        plt.title('MAE: {}'.format(mae))
        plt.ylabel('MAE')
        plt.xlabel('Time [h]')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(preds2[i, :, -1], label='prediction')
        plt.plot(trues2[i, :, -1], label='ground truth')
        mae = np.mean(np.abs(preds2[i, :, -1] - trues2[i, :, -1]))
        plt.title('MAE: {}'.format(mae))
        plt.legend()
        plt.show()

def visualize_prediction_std_sort(preds, trues, prev_seqs, base_dir):
    """
    Visualize the prediction and actual values.
    """
    stds = []
    for i in tqdm(range(preds.shape[0])):
        std = np.std(trues[i, :, -1])
        stds.append(std)
    stds_sorted = np.sort(stds)
    # print(stds_sorted)
    # 1/4 of the data
    thr1 = stds_sorted[int(len(stds_sorted)/4)]
    thr2 = stds_sorted[int(len(stds_sorted)/4*2)]
    thr3 = stds_sorted[int(len(stds_sorted)/4*3)]

    std1_dir = os.path.join(base_dir, 'std1')
    if not os.path.exists(std1_dir):
        os.makedirs(std1_dir)
    std2_dir = os.path.join(base_dir, 'std2')
    if not os.path.exists(std2_dir):
        os.makedirs(std2_dir)
    std3_dir = os.path.join(base_dir, 'std3')
    if not os.path.exists(std3_dir):
        os.makedirs(std3_dir)
    std4_dir = os.path.join(base_dir, 'std4')
    if not os.path.exists(std4_dir):
        os.makedirs(std4_dir)

    basetime = datetime.datetime(2016, 1, 1, 0, 0, 0)
    plt.figure()

    preds = np.concatenate((prev_seqs, preds), axis=1)
    trues = np.concatenate((prev_seqs, trues), axis=1)

    for i in tqdm(range(preds.shape[0])):
        offset = - prev_seqs.shape[1] + 1
        x = np.arange(offset, preds.shape[1] + offset)
        mae = np.mean(np.abs(preds[i, prev_seqs.shape[1]:, -1] - trues[i, prev_seqs.shape[1]:, -1]))
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.plot(x, trues[i, :, -1], label='(a)', color='blue', zorder=2)
        plt.plot(x, preds[i, :, -1], label='(b)', color='red', zorder=1)
        current_time = basetime + datetime.timedelta(hours=i)
        plt.title(f'MAE: {mae:.3f}, std: {stds[i]:.3f}, {current_time}')
        # x label

        plt.ylabel('logXmax1h')
        plt.xlabel('Time [h]')
        plt.axvline(x=0, color='black', linestyle='--', zorder=0)

        plt.ylim(-3, 5)
        plt.legend()
        if np.std(trues[i, :, -1]) < thr1:
            plt.savefig(f'{std1_dir}/{mae:.3f}_{i:04}.png')
        elif np.std(trues[i, :, -1]) < thr2 and np.std(trues[i, :, -1]) >= thr1:
            plt.savefig(f'{std2_dir}/{mae:.3f}_{i:04}.png')
        elif np.std(trues[i, :, -1]) < thr3 and np.std(trues[i, :, -1]) >= thr2:
            plt.savefig(f'{std3_dir}/{mae:.3f}_{i:04}.png')
        else:
            plt.savefig(f'{std4_dir}/{mae:.3f}_{i:04}.png')
        plt.cla()


if __name__ == "__main__":

    base_dir = 'results/FT_MAE_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_ffill_2016_0'
    image_dir = base_dir.replace('results', 'result_images')
    preds = np.load(os.path.join(base_dir, 'pred.npy'))
    trues = np.load(os.path.join(base_dir, 'true.npy'))
    metrics = np.load(os.path.join(base_dir, 'metrics.npy'))
    prev_seqs = np.load(os.path.join(base_dir, 'prev_seqs.npy'))
    
    base_dir2 = 'results/FT_Flare_stddev_ftMS_sl8_ll2_pl24_dm128_nh8_el1_dl1_df16_atfull_fc5_ebtimeF_dtTrue_mxTrue_ffill_2016_0'
    preds2 = np.load(os.path.join(base_dir2, 'pred.npy'))
    trues2 = np.load(os.path.join(base_dir2, 'true.npy'))
    metrics2 = np.load(os.path.join(base_dir2, 'metrics.npy'))

    image_dir = base_dir.replace('results', 'result_images')

    print(trues.shape)
    print(preds.shape)
    print(metrics)
    # visualize_prediction_sample(preds, trues, image_dir)
    # visualize_predictions(preds, trues, preds2, trues2)
    # visualize_predictions_compare(preds, trues, preds2, trues2, image_dir)
    # visualize_prediction_std_sort(preds, trues, prev_seqs, image_dir)
    visualize_prediction(preds, trues, prev_seqs, image_dir)

