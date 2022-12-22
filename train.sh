poetry run python src/flareformer/main.py --params config/params_2018_replace.json --warmup_epochs=5 --epoch_for_2stage=3 --imbalance --wandb --model_name replace18_dropout095
poetry run python src/flareformer/main.py --params config/params_2019_replace.json --warmup_epochs=5 --epoch_for_2stage=3 --imbalance --wandb --model_name replace19_dropout095
poetry run python src/flareformer/main.py --params config/params_2020_replace.json --warmup_epochs=5 --epoch_for_2stage=3 --imbalance --wandb --model_name replace20_dropout095

# poetry run python src/flareformer/main.py --params config/params_2018_replace_weight.json --warmup_epochs=5 --epoch_for_2stage=3 --imbalance --wandb --model_name replace18_GMGS_1000
# poetry run python src/flareformer/main.py --params config/params_2019_replace_weight.json --warmup_epochs=5 --epoch_for_2stage=3 --imbalance --wandb --model_name replace19_GMGS_1000
# poetry run python src/flareformer/main.py --params config/params_2020_replace_weight.json --warmup_epochs=5 --epoch_for_2stage=3 --imbalance --wandb --model_name replace20_GMGS_1000