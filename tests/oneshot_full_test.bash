#!/bin/bash -eu
path="data/ft_database_all17.jsonl"
args=""
for i in `seq 1 4`
do
    jsonl=$(cat $path | head -n $i | tail -n 1)
    img_path=$(echo $jsonl | jq -r ".magnetogram")
    feature=$(echo $jsonl | jq ".feature")
    args=" -F 'image_feats=@${img_path}' -F 'physical_feats=${feature}' ${args}"
    # echo $args
done

com="curl -X POST ${args} http://127.0.0.1:8080/oneshot/full"

# echo $com
eval $com
