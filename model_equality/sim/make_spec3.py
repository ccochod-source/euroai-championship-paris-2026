"""Jeu n° 3 : exemples étiquetés pour le classifieur de paires + réservoirs d'échantillons locaux pour RUT.
4 familles (Llama-3.2-1B, Qwen2.5-3B, Qwen2.5-1.5B, et SmolLM2-360M pour valider sur les vrais refs).
Par famille :
  - réservoirs RUT (prompts uniques x M échantillons) : bf16, fp8w, int8w, fp16 ;
  - instances « endpoint » (protocole complet, K blocs chacune) : déguisements de S et variantes de poids.
Usage : python make_spec3.py [T_plain]   (défaut 0.7)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
M = int(os.environ.get("RUT_M", "64")); K = int(os.environ.get("CLF_K", "6"))
SYS = [
    "You are a helpful assistant for general questions and everyday tasks.\nAsk a clarifying question when the request is ambiguous.\nDo not make up facts; say when you do not know.\nUse short paragraphs.\nDo not share personal data about private individuals.",
    "Help the user with their questions and tasks. State the main point first, then add detail. Ask a clarifying question when the request is ambiguous. Use a warm, friendly tone.",
    "You are a helpful customer-support assistant for an online bookstore called PageTurner. Keep answers clear, friendly and concise.",
    "You are a helpful assistant. Answer as concisely as possible.",
    "You are LexAssist, the virtual assistant of a law firm. You provide general legal information only and never give medical advice.",
    "Always answer formally and politely, in complete sentences. Never use emojis or slang.",
]
PERS = [
    "You are an etymologist. Explain the origin and history of words whenever relevant.",
    "You are a text-based video game console. Respond as the game would.",
    "You are Emoji Artist. Express ideas with emojis as much as possible.",
    "You are a basketball coach who motivates and trains players.",
    "You are GolfGPT, an expert golf instructor.",
    "You are a Research Paper Evaluator who assesses academic writing critically.",
    "You are Claude, an AI assistant created by Anthropic. You are thoughtful, honest and helpful.",
    "You are Captain Remy, a cheerful retired sea captain who loves telling stories. Stay in character.",
    "You are Professor Hawthorne, a slightly absent-minded but brilliant historian. Stay in character.",
    "You are Zed, a sarcastic teenage hacker from the year 2099. Stay in character.",
    "You are Sister Agnes, a kind and patient medieval nun. Stay in character.",
    "You are Chef Marco, a passionate and grumpy Italian chef. Stay in character.",
    "You are ChatGPT, a large language model trained by OpenAI.",
    "You are Aria, the friendly assistant of TechNova Inc.",
]
FAM = {"LL1": ("unsloth/Llama-3.2-1B-Instruct", "ll1"), "QA3": ("Qwen/Qwen2.5-3B-Instruct", "q3"),
       "QB15": ("Qwen/Qwen2.5-1.5B-Instruct", "q15"), "S360": ("HuggingFaceTB/SmolLM2-360M-Instruct", "s360")}
c = dict(temp=TP, top_p=1.0, top_k=0, rep_pen=1.0)
eps = []
for fam, (base, tag) in FAM.items():
    def add(name, model, weights, **kw):
        cfg = dict(c); cfg.update(kw)
        eps.append(dict(ep=f"{fam}-{name}", family=fam, model=f"{fam}_{model}", variant=name, base=base, weights=weights, **cfg))
    # réservoirs RUT
    for w in ("bf16", "fp8w", "int8w", "fp16"):
        add(f"pool_{w}", {"bf16": "S", "fp16": "fp16"}.get(w, w.replace("w", "")), w, uniq=True, reps=M if w != "fp16" else M // 2)
    # déguisements de S
    add("plain", "S", "bf16", reps=K)
    for i, sp in enumerate(SYS):
        add(f"sys{i}", "S", "bf16", system=sp, reps=K // 2)
    add("persona_rand", "S", "bf16", system=PERS, reps=K)
    add("persona_claude", "S", "bf16", system=PERS[0], reps=K // 2)
    add("persona_remy", "S", "bf16", system=PERS[1], reps=K // 2)
    add("wm_kgw", "S", "bf16", watermark="kgw", reps=K)
    add("wm_selfhash", "S", "bf16", watermark="kgw_selfhash", reps=K // 2)
    add("T1", "S", "bf16", temp=1.0, reps=K)
    # variantes de poids (modèles différents)
    add("fp8", "fp8", "fp8w", reps=K)
    add("fp8a", "fp8", "fp8wa", reps=K // 2)
    add("int8", "int8", "int8w", reps=K)
    add("fp16", "fp16", "fp16", reps=K // 2)
    add("nf4", "nf4", "nf4", reps=K // 2)
    # variantes servies avec un échantillonneur tronqué (vu sur ep-27 / ep-14 / ep-07 : top-k 50, T plus haute)
    add("int8_trunc", "int8t", "int8w", temp=1.0, top_k=50, top_p=0.95, reps=K // 2)
    add("nf4_trunc", "nf4t", "nf4", temp=1.0, top_k=50, reps=K // 2)
    add("lora_med_0.2_trunc", "Lmed0.2t", f"lora:out/lora/{tag}_med:0.2", temp=0.9, top_k=50, top_p=0.95, reps=K // 2)
    for ad in ("alpaca", "alp2", "med"):
        for st in ("0.1", "0.2", "0.4", "0.8"):   # 0.1 / 0.2 / 0.4 reproduisent l'excès des LoRA 0.3 / 0.7 / 1.5 des refs
            add(f"lora_{ad}_{st}", f"L{ad}{st}", f"lora:out/lora/{tag}_{ad}:{st}", reps=K)
    # déguisement appliqué à une variante (persona sur fp8, system prompt sur LoRA)
    add("fp8_persona", "fp8", "fp8w", system=PERS[0], reps=K // 2)
    add("lora_alpaca_0.2_sys", "Lalpaca0.2", f"lora:out/lora/{tag}_alpaca:0.2", system=SYS[0], reps=K // 2)
json.dump({"endpoints": eps, "T_plain": TP}, open(os.path.join(HERE, "spec_set3.json"), "w"), indent=1)
n = sum((89 if e.get("uniq") else 110) * e["reps"] for e in eps) // len(FAM)
print(len(eps), "configurations ;", n, "générations par famille")
