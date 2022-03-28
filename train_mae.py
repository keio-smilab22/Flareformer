from mae.prod.train import *
from mae.prod.preprocess import *
import json
from src.Dataloader import TrainDataloader256

if __name__ == '__main__':
    # Preprocess().run()
     
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    params = json.loads(open("params/params_2014.json").read())
    train_dataset = TrainDataloader256("train", params["dataset"],has_window=False)
    # aug_train_dataset = TrainDataloader("train", params["dataset"], augmentation=True)
    mean, std = train_dataset.calc_mean()
    print(mean, std)
    train_dataset.set_mean(mean, std)

    main(args,train_dataset)
