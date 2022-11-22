# python main_informer.py --model FT --data Flare --des open_test --batch_size 64 --feat_path data_all_v2_2.csv

# python main_informer.py --model FT --data Flare --des open_test_ffill --batch_size 64 --feat_path data_all_v2_ffill.csv --lradj none --train_epochs 40 --learning_rate 1e-6

# python main_informer.py --model FT --data Flare --des open_test_bfill --batch_size 64 --feat_path data_all_v2_bfill.csv --lradj none --train_epochs 40 --learning_rate 1e-6

# python main_informer.py --model FT --data Flare --des open_test --batch_size 64 --feat_path data_all_v2.csv --lradj none --train_epochs 40 --learning_rate 1e-6

# python main_informer.py --model FT --data Flare --des open_test_fillna0 --batch_size 64 --feat_path data_all_v2_fillna0.csv --lradj none --train_epochs 40 --learning_rate 1e-6


# python main_informer.py --model FT --data Flare --des open_test_interp_time --batch_size 64 --feat_path data_all_v2_interp_time.csv --learning_rate 1e-6 --train_epochs 40

# python main_informer.py --model FT --data Flare --des open_test_interp_time_adjnone --batch_size 64 --feat_path data_all_v2_interp_time.csv --learning_rate 1e-6 --train_epochs 40 --lradj none


# python main_informer.py --model FT --data Flare --des closed_test_fillna0 --batch_size 64 --feat_path data_all_v2_fillna0.csv

# python main_informer.py --model FT --data Flare --des closed_test_ffill --batch_size 64 --feat_path data_all_v2_ffill.csv

# python main_informer.py --model FT --data Flare --des closed_test_bfill --batch_size 64 --feat_path data_all_v2_bfill.csv

# NOTE 1e-6 adj none fillna0
# python main_informer.py --model FT_MAE --data Flare --batch_size 8 --feat_path data_all_v2_fillna0.csv --learning_rate 1e-6 --train_epochs 40 --lradj none

# python main_informer.py --model FT_MAE_linear --data Flare --batch_size 8 --feat_path data_all_v2_fillna0.csv --learning_rate 1e-6 --train_epochs 40 --lradj none

# python main_informer.py --model FT_linear --data Flare --batch_size 64 --feat_path data_all_v2_fillna0.csv --learning_rate 1e-6 --train_epochs 40 --lradj none

# for method in ffill
# do
#     for y in 2017 2016 2015
#     do
#     python main_informer.py --model FT --data Flare_stddev --des ${method}_${y} --batch_size 64 --feat_path data_all_stddev_${method}.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y} --seq_len 8
#     done
# done

# for method in ffill
# do
#     for y in 2017 2016 2015
#     do
#     python main_informer.py --model FT_linear --data Flare_stddev --des ${method}_${y} --batch_size 64 --feat_path data_all_stddev_${method}.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y} --seq_len 8
#     done
# done

# for method in ffill
# do
#     for y in 2017 2016 2015
#     do
#     python main_informer.py --model FT --data Flare_stddev --des ${method}_${y} --batch_size 64 --feat_path data_all_stddev_${method}.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y} --seq_len 8 --attn full
#     done
# done

# for method in ffill
# do
#     for y in 2017 2016 2015
#     do
#     python main_informer.py --model FT_linear --data Flare_stddev --des ${method}_${y} --batch_size 64 --feat_path data_all_stddev_${method}.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y} --seq_len 8 --attn full
#     done
# done

# for seq_len in 4
# do

#     for y in 2017 2016 2015
#     do

#     label_len=$((${seq_len}/2))

#     python main_informer.py --model FT --data Flare_stddev --des ${y}_ft_corrected --batch_size 64 --feat_path data_all_stddev_ffill.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y} --seq_len ${seq_len} --label_len ${label_len} --attn full
#     done
# done


# for l in 2 3 1
# do
#     for y in 2015
#     do
#     python main_informer.py --model FT_MAE --data Flare_stddev --des lays_${y} --batch_size 8 --feat_path data_all_stddev_ffill.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y} --e_layers ${l} --d_layers ${l}
#     done
# done

# python main_informer.py --model FT --data Flare_sunpy --des sunpy --batch_size 4 --lradj none --train_epochs 20 --learning_rate 1e-6 --year 2012 --seq_len 12 --label_len 6 --attn full --root_path ./data/noaa --features S --pred_len 24

# binary search for gamma from 0.01 to 100


for gamma in 1
do
    python main_informer.py \
    --rbf_gamma ${gamma} \
    --model FT \
    --data Flare_sunpy \
    --des gmgs6_${gamma}_notOC100 \
    --batch_size 32 \
    --lradj none \
    --train_epochs 20 \
    --learning_rate 1e-6 \
    --year 2020 \
    --seq_len 12 \
    --label_len 6 \
    --attn full \
    --root_path ./data/noaa \
    --features S \
    --pred_len 48 \
    --loss gmgs6 \
    --use_amp \
    --score_matrix_1 0.354998803807886 -0.06988773489386695 -60.53181202862116 -100 \
    --score_matrix_2 -0.06988773489386695 1.4770689845785805 94.16385991862359 54.69567194724475 \
    --score_matrix_3 -60.53181202862116 94.16385991862359 182.47627878491521 143.00809081353638 \
    --score_matrix_4 -100 54.69567194724475 143.00809081353638 357.4559210121284
done

for gamma in 1
do
    python main_informer.py \
    --rbf_gamma ${gamma} \
    --model FT \
    --data Flare_sunpy \
    --des gmgs6_${gamma}_predMX100k \
    --batch_size 32 \
    --lradj none \
    --train_epochs 20 \
    --learning_rate 1e-6 \
    --year 2020 \
    --seq_len 12 \
    --label_len 6 \
    --attn full \
    --root_path ./data/noaa \
    --features S \
    --pred_len 48 \
    --loss gmgs6 \
    --use_amp \
    --score_matrix_1 0.354998803807886 -0.06988773489386695 -60531.81202862116 -100000 \
    --score_matrix_2 -0.06988773489386695 1.4770689845785805 94163.85991862359 54695.67194724475 \
    --score_matrix_3 -0.6053181202862116 0.9416385991862359 182476.27878491521 143008.09081353638 \
    --score_matrix_4 -1.0 0.5469567194724475 143008.09081353638 357455.9210121284
done

# [0.354998803807886, -0.06988773489386695, -0.6053181202862116, -1.0],
# [
#     -0.06988773489386695,
#     1.4770689845785805,
#     0.9416385991862359,
#     0.5469567194724475,
# ],
# [
#     -0.6053181202862116,
#     0.9416385991862359,
#     1.8247627878491521,
#     1.4300809081353638,
# ],
# [-1.0, 0.5469567194724475, 1.4300809081353638, 3.574559210121284]

# python main_informer.py --model FT --data Flare_sunpy --des rbf_10 --batch_size 32 --lradj none --train_epochs 20 --learning_rate 1e-6 --year 2018 --seq_len 12 --label_len 6 --attn full --root_path ./data/noaa --features S --pred_len 48

