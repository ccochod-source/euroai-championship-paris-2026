for x in ref-07 ref-04 ref-03 ref-09; do
  python distill.py --model HuggingFaceTB/SmolLM2-360M-Instruct --raw data/raw --train-ep $x --eval-eps ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09 --out out/distill/real_$x.json
done
