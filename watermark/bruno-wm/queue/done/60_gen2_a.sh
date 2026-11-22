export OMP_NUM_THREADS=8; python gen_clean2.py && (python gen_wm2.py trustmark --indomain; python gen_wm2.py trustmark)
