from argparse import Namespace
from torch.utils.data import DataLoader
from datasets.flare import FlareDataset, ForecastDataset
from datasets.sampler import TrainBalancedBatchSampler


def load_datasets(args: Namespace, debug: bool):
    if args.stage == 'flareformer':
        train_dataset = FlareDataset("train", args.dataset, debug=debug) # (48617, 1, 256, 256)
        val_dataset = FlareDataset("valid", args.dataset, debug=debug) # (7991, 1, 256, 256)
        test_dataset = FlareDataset("test", args.dataset, debug=debug)
    elif args.stage == 'forecast':
        train_dataset = ForecastDataset("train", args.dataset, debug=debug) # (48617, 1, 256, 256)
        val_dataset = ForecastDataset("valid", args.dataset, debug=debug) # (7991, 1, 256, 256)
        test_dataset = ForecastDataset("test", args.dataset, debug=debug)
    else:
        print("No dataset")
    mean, std = train_dataset.calc_mean()

    train_dataset.set_mean(mean, std)
    val_dataset.set_mean(mean, std)
    test_dataset.set_mean(mean, std)

    return train_dataset, val_dataset, test_dataset


def prepare_dataloaders(args: Namespace, debug: bool, imbalance: bool):
    print("Prepare Dataloaders")
    train_dataset, val_dataset, test_dataset = load_datasets(args, debug)

    if imbalance:
        train_dl = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    else:
        batch_sampler = TrainBalancedBatchSampler(train_dataset, args.output_channel, args.bs // args.output_channel)
        train_dl = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2)

    val_dl = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    sample = train_dataset[0]

    return (train_dl, val_dl, test_dl), sample
