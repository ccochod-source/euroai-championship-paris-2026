#!/bin/bash
# wait for the two TrustMark train shards, merge them, then run the test shards (whole-frame test images)
cd /p/scratch/training2666/dasilvalopes3/wm/s3
until grep -q "^== trustmark train" gap/logs/sq_tm_train0.log && grep -q "^== trustmark train" gap/logs/sq_tm_train1.log; do sleep 10; done
source /p/scratch/training2666/dasilvalopes3/wm/kit/.venv/bin/activate
python - <<"PY"
import json
d = {}
for k in (0, 1): d.update(json.load(open(f"out/frame_trustmark_train.{k}of2.json")))
json.dump(d, open("out/frame_trustmark_train.json", "w")); print("merged", len(d))
PY
(./sg 0 gap/sqsearch.py trustmark test 0 0/2 > gap/logs/sq_tm_test0.log 2>&1 &)
./sg 1 gap/sqsearch.py trustmark test 0 1/2 > gap/logs/sq_tm_test1.log 2>&1
until grep -q "^== trustmark test" gap/logs/sq_tm_test0.log; do sleep 10; done
python - <<"PY"
import json
d = {}
for k in (0, 1): d.update(json.load(open(f"out/frame_trustmark_test.{k}of2.json")))
json.dump(d, open("out/frame_trustmark_test.json", "w")); print("merged", len(d))
PY
echo CHAIN_DONE
