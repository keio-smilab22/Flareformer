for i in 2 3
do
    for y in `seq 2014 2017`
    do
        cat params_${y}.json | sed "s/\"N\": 1/\"N\": ${i}/g" > paramsN${i}_${y}.json
    done
done