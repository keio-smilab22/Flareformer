from torch.utils.data import DataLoader
from datasets.flare import FlareDataset
from datasets.sampler import TrainBalancedBatchSampler


def load_datasets(args):
    train_dataset = FlareDataset("train", args.dataset)
    val_dataset = FlareDataset("valid", args.dataset)
    test_dataset = FlareDataset("test", args.dataset)

    mean, std = train_dataset.calc_mean()
    print(f"(mean,std) = ({mean},{std})")

    train_dataset.set_mean(mean, std)
    val_dataset.set_mean(mean, std)
    test_dataset.set_mean(mean, std)

    return train_dataset, val_dataset, test_dataset


def prepare_dataloaders(args, imbalance):
    train_dataset, val_dataset, test_dataset = load_datasets(args)

    if imbalance:
        train_dl = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    else:
        batch_sampler = TrainBalancedBatchSampler(train_dataset, args.output_channel, args.bs // args.output_channel)
        train_dl = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2)

    val_dl = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)
    sample = train_dataset[0]

    return (train_dl, val_dl, test_dl), sample
