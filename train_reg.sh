# for i in 0 1 2 3 4
# do
#     python main_informer.py --model regression --data Flare --label_len ${i} --des label_len${i}_mae --batch_size 8
# done

# python main_informer.py --model FT --data Flare --des open_test --batch_size 64 --feat_path data_all_v2_2.csv

python main_informer.py --model FT --data Flare --des open_test_ffill --batch_size 64 --feat_path data_all_v2_ffill.csv --lradj none --train_epochs 20 --learning_rate 1e-5

python main_informer.py --model FT --data Flare --des open_test_ffill --batch_size 64 --feat_path data_all_v2_bfill.csv --lradj none --train_epochs 20 --learning_rate 1e-5

python main_informer.py --model FT --data Flare --des open_test --batch_size 64 --feat_path data_all_v2.csv --lradj none --train_epochs 20 --learning_rate 1e-5

python main_informer.py --model FT --data Flare --des open_test_fillna0 --batch_size 64 --feat_path data_all_v2_fillna0.csv --lradj none --train_epochs 20 --learning_rate 1e-5


# python main_informer.py --model FT --data Flare --des closed_test_interp_time --batch_size 64 --feat_path data_all_v2_interp_time.csv --learning_rate

# python main_informer.py --model FT --data Flare --des closed_test_fillna0 --batch_size 64 --feat_path data_all_v2_fillna0.csv

# python main_informer.py --model FT --data Flare --des closed_test_ffill --batch_size 64 --feat_path data_all_v2_ffill.csv

# python main_informer.py --model FT --data Flare --des closed_test_bfill --batch_size 64 --feat_path data_all_v2_bfill.csv