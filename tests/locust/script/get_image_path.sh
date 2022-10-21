# set test name
test_name="get_image_path"

# get date now
test_date=`date "+%Y%m%d-%H%M%S"`

# set path
root_dir=$(cd `dirname $0`/../../../; pwd)
config_dir=${root_dir}/tests/locust/config
result_dir=$root_dir/tests/locust/result/${test_date}_${test_name}

# make result dir
mkdir -p $result_dir

# start server 
gnome-terminal -- bash -c "poetry run python src/flareformer/main.py --mode=server --params=config/params_2017.json > ${result_dir}/server.log ; exit"

# wait server starting
echo "** wait server starting (10 sec) **"
sleep 10
echo "** locust test start! **"

# locust test run
poetry run locust --config ${config_dir}/${test_name}.conf --html ${result_dir}/locust.html

# kill gnome-terminal
pkill -f "poetry run python src/flareformer/main.py"
