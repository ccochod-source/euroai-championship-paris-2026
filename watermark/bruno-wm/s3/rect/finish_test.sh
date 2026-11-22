#!/bin/bash
# CPU only, runs on the login node. Waits (at most 60 min) for the two test runs started at 20:04 on 21 Nov
# (rect/log_ps_test2.txt, rect/log_ss_test2.txt) and then writes the final rects_<method>_test.json files.
# It starts nothing on the GPU. If a run died, it does nothing for that method: resume by hand (see RESULTS.md).
cd /p/scratch/training2666/dasilvalopes3/wm/s3 || exit 1
export PYTHONPATH=.:../kit PYTHONWARNINGS=ignore
PY=../kit/.venv/bin/python; ss=0; ps=0
for i in $(seq 1 180); do
  if [ $ss = 0 ] && grep -q "DONE rect_raw2_stegastamp_test.json" rect/log_ss_test2.txt 2>/dev/null; then
    $PY rect/finalize.py stegastamp test rect_raw2_stegastamp_test.json --thr 0.90 --variant hc --shrink 3 >> rect/log_finish.txt 2>&1; ss=1
  fi
  if [ $ps = 0 ] && grep -q "DONE rect_raw_pixelseal_test.json" rect/log_ps_test2.txt 2>/dev/null; then
    $PY rect/finalize.py pixelseal test rect_raw_pixelseal_test.json --thr 0.70 >> rect/log_finish.txt 2>&1; ps=1
  fi
  [ $ss = 1 ] && [ $ps = 1 ] && break
  sleep 20
done
echo "finish_test.sh exit: stegastamp=$ss pixelseal=$ps $(date)" >> rect/log_finish.txt
