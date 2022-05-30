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

for method in ffill time fillna0
do
    for y in 2015 2016 2017
    do
    python main_informer.py --model FT --data Flare_stddev --des ${method} --batch_size 64 --feat_path data_all_stddev_${method}.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y}
    python main_informer.py --model FT_linear --data Flare_stddev --des ${method} --batch_size 64 --feat_path data_all_stddev_${method}.csv --magnetogram_path data_magnetogram_256_stddev.npy --lradj none --train_epochs 20 --learning_rate 1e-6 --year ${y}
    done
done
