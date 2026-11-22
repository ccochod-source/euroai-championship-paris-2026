"""Construit sim/protocol.jsonl : toutes les sondes de l'équipe (v1 + v2a ep + lots refs en file), version ref-01/ep-01."""
import glob, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
items, seen = [], set()
def add(pid, rep, prompt, mt, src):
    if (pid, rep) in seen: return
    seen.add((pid, rep)); items.append({"prompt_id": pid, "repeat": rep, "prompt": prompt, "max_tokens": mt or 256, "src": src})
for l in open(os.path.join(ROOT, "data/raw/ep-01.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r["prompt_id"].startswith("v1_"): add(r["prompt_id"], r["repeat"], r["prompt"], 256, "v1")
for d in ("pending", "done"):
    for p in sorted(glob.glob(os.path.join(ROOT, "queue", d, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        for l in open(p, encoding="utf-8"):
            if not l.strip(): continue
            j = json.loads(l)
            if j["endpoint"] in ("ep-01", "ref-01"):
                add(j["prompt_id"], int(j["repeat"]), j["prompt"], j.get("max_tokens"), name)
with open(os.path.join(ROOT, "sim/protocol.jsonl"), "w", encoding="utf-8") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
import collections
print(len(items), collections.Counter(i["src"] for i in items))
