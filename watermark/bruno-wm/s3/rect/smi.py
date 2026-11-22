import subprocess
print(subprocess.run("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; ps -u $USER -o pid,etimes,pcpu,args | grep -E \"rect/\" | grep -v grep | cut -c1-120", shell=True, capture_output=True, text=True).stdout)
