import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(preds, trues):
    """
    Visualize the prediction and actual values.
    """
    plt.figure()
    
    for i in range(preds.shape[0] // 400):
        plt.plot(preds[i, :, -1], label='prediction')
        plt.plot(trues[i, :, -1], label='ground truth')
        plt.legend()
        plt.show()
    # q


if __name__ == "__main__":
    preds = np.load('results/FT_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_fillna0_0/pred.npy')

    trues = np.load('results/FT_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_fillna0_0/true.npy')

    metrics = np.load('results/FT_Flare_stddev_ftMS_sl4_ll2_pl24_dm128_nh8_el1_dl1_df16_atprob_fc5_ebtimeF_dtTrue_mxTrue_fillna0_0/metrics.npy')   

    print(trues.shape)
    print(preds.shape)
    print(metrics)
    visualize_prediction(preds, trues)