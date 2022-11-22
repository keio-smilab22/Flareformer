python main_informer_test.py \
--model FT \
--data Flare_sunpy \
--des gmgs6_1_notOC100 \
--batch_size 32 \
--lradj none \
--train_epochs 20 \
--learning_rate 1e-6 \
--year 2014 \
--seq_len 12 \
--label_len 6 \
--attn full \
--root_path ./data/noaa \
--features S \
--pred_len 48