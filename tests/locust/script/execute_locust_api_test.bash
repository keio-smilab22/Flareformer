#!/bin/bash -eu
# check argument
if [ $# -ne 1 ]; then
  echo "You must specify the test name as the 1st argument."
  exit 1
fi

# set test name
test_name=$1

# get date now
test_date=$(date "+%Y%m%d-%H%M%S")

# set path
root_dir=$(readlink -f $(dirname $0)/../../../)
script_dir=${root_dir}/tests/locust/script
result_dir=${root_dir}/tests/locust/result/${test_date}_${test_name}
config_path=${root_dir}/tests/locust/config/locust.conf

# check test name
if [ ! -e  "${script_dir}/${test_name}.py" ]; then
  echo "Invalid test name. The config file for locust was not found.: ${script_dir}/${test_name}.py"
  exit 2
fi

# change directory to repository root
cd $root_dir

# make result dir
mkdir -p $result_dir

# start server
gnome-terminal -- bash -c "poetry run python src/flareformer/main.py --mode=server --params=config/params_2017.json > ${result_dir}/server.log ; exit"

# wait server starting
echo "** wait server starting (10 sec) **"
sleep 10
echo "** locust test start! (60 sec) **"

# locust test run
poetry run locust --config ${config_path} --locustfile ${script_dir}/${test_name}.py --html ${result_dir}/locust.html 2> ${result_dir}/console.log

# kill gnome-terminal
pkill -f "poetry run python src/flareformer/main.py"
echo "** locust test finished!! **"
