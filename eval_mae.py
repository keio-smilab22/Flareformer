from mae.prod.eval import *
from train_mae import FashionMnistDataLoader
import argparse
import json
from mae.prod.datasets import *
import mae.prod.eval_seq as seq

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument(
        '--batch_size',
        default=130,
        type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--baseline', default="attn")
    parser.add_argument('--enc_depth', default=12, type=int)
    parser.add_argument('--dec_depth', default=8, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--input_size', default=256, type=int,  # 512の場合はここ変える
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument(
        '--model',
        default='vit_for_FT',
        type=str,
        metavar='MODEL',
        help='Name of model to train')

    parser.add_argument(
        '--norm_pix_loss',
        action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',  # 0.0000007
                        help='learning rate (absolute lr)')
    parser.add_argument(
        '--blr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='../../workspace/flare_transformer/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--batch_size_search', default=False, action='store_true')

    parser.add_argument('--target', default="m") # m, p
    parser.add_argument('--token_window', default=4, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.set_defaults(pin_mem=True)

    return parser

parser = get_args_parser()
args = parser.parse_args()

if args.checkpoint is None:
    args.checkpoint = f'/home/initial/Dropbox/flare_transformer/output_dir/{args.baseline}/checkpoint-2.pth'

if args.target == "seq":
    seq.run(args)
else:
    # dl = TrainDataloader()
    # img, _ = dl[0]
    # img = img.transpose((1, 2, 0))

    params = json.loads(open("params/params_2014.json").read())
    dl = TrainDataloader256("train", params["dataset"],has_window=False)
    mean,std = dl.calc_mean()
    dl.set_mean(mean,std)
    print(mean,std)

    dl2 = TrainDataloader256("test", params["dataset"],has_window=False)
    dl2.set_mean(mean,std)


    img, _ = dl[0]
    img_test, _ = dl2[1]
    img = img.transpose(0,1).transpose(1,2)
    img_test = img_test.transpose(0,1).transpose(1,2)

    # dl = FashionMnistDataLoader()
    # img, _, _ = dl[0]
    # print(img.shape)
    # img = img.transpose(0,1).transpose(1,2)
    # print(img.shape)

    # assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    # img = img - imagenet_mean
    # img = img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(torch.tensor(img))

    chkpt_dir = args.checkpoint

    model_mae = prepare_model(chkpt_dir,img_size=args.input_size,baseline=args.baseline,embed_dim=128)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    # run_one_image(img, model_mae, mean,std)
    run_one_image(img_test, model_mae, mean,std)