#!/usr/bin/env bash
# Lance le scoreur CPU en arrière-plan avec le garde-fou RAM.  usage : ./cpu_run.sh <log> <args de cpu_lik.py...>
cd "$(dirname "$0")"
export HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false SIM_THREADS=${SIM_THREADS:-4}
LOG=$1; shift
nohup nice -n 5 ./venv_cpu/bin/python cpu_lik.py "$@" >> "logs/$LOG" 2>&1 &
echo $! > logs/cpu.pid
nohup ./cpu_guard.sh >/dev/null 2>&1 &
