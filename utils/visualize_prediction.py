import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


def visualize_prediction(preds, trues, base_dir):
    """
    Visualize the prediction and actual values.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # os.makedirs('result_images/FT_MAE_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_ffill_2016_0', exist_ok=True)
    
    min_mae = np.inf
    for i in tqdm(range(preds.shape[0])):
        plt.figure()
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.plot(trues[i, :, -1], label='(a)', color='blue')
        plt.plot(preds[i, :, -1], label='(b)', color='red')
        mae = np.mean(np.abs(preds[i, :, -1] - trues[i, :, -1]))
        if mae < min_mae:
            min_mae = mae
            min_mae_idx = i

        plt.title(f'MAE: {mae:.3f}')
        plt.ylabel('logXmax1h')
        plt.xlabel('Time [h]')
        plt.ylim(-2.0, 6.0)
        plt.legend()
        plt.savefig(f'{base_dir}/{mae:.3f}_{i:04}.png')
        plt.close()
    print(f'Min MAE: {min_mae} at {min_mae_idx}')
    # q


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


if __name__ == "__main__":

    base_dir = 'results/FT_MAE_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_ffill_2016_0'
    image_dir = base_dir.replace('results', 'result_images')
    preds = np.load(os.path.join(base_dir, 'pred.npy'))
    trues = np.load(os.path.join(base_dir, 'true.npy'))
    metrics = np.load(os.path.join(base_dir, 'metrics.npy'))
    
    # base_dir2 = 'results/FT_MAE_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_ffill_2015_0'
    # preds2 = np.load(os.path.join(base_dir2, 'pred.npy'))
    # trues2 = np.load(os.path.join(base_dir2, 'true.npy'))
    # metrics2 = np.load(os.path.join(base_dir2, 'metrics.npy'))

    print(trues.shape)
    print(preds.shape)
    print(metrics)
    visualize_prediction(preds, trues, image_dir)
    # visualize_predictions(preds, trues, preds2, trues2)