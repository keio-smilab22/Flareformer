#!/bin/bash -eu
# check argument
if [ $# -ne 1 ]; then
  echo "Set the test name as 1 argument"
  exit 
fi

# set test name
test_name=$1

# set path
root_dir=$(readlink -f .)
config_dir=${root_dir}/tests/locust/config
script_dir=${root_dir}/tests/locust/script
result_dir=$root_dir/tests/locust/result/${test_date}_${test_name}

# check test name
if [ ! -e  "${script_dir}/${test_name}.py" ]; then
  echo "Test name is not correct."
  exit
fi

# get date now
test_date=$(date "+%Y%m%d-%H%M%S")

# make result dir
mkdir -p $result_dir

# start server 
gnome-terminal -- bash -c "cd ${root_dir} | poetry run python src/flareformer/main.py --mode=server --params=config/params_2017.json > ${result_dir}/server.log ; exit"

# wait server starting
echo "** wait server starting (10 sec) **"
sleep 10
echo "** locust test start! **"

# locust test run
poetry run locust --config ${config_dir}/${test_name}.conf -f ${script_dir}/${test_name}.py --html ${result_dir}/locust.html 2> ${result_dir}/console.log

# kill gnome-terminal
pkill -f "poetry run python src/flareformer/main.py"
echo "** locust test finished!! **"
