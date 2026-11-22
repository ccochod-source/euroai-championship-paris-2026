#!/usr/bin/env bash
# Garde-fou RAM : arrête le scoreur CPU (PID dans logs/cpu.pid) si la mémoire disponible passe sous 350 Mo.
while kill -0 "$(cat logs/cpu.pid)" 2>/dev/null; do
  av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$av" -lt 350 ]; then kill "$(cat logs/cpu.pid)"; echo "$(date +%T) ARRÊT : RAM disponible $av Mo" >> logs/cpu_guard.log; fi
  sleep 5
done
