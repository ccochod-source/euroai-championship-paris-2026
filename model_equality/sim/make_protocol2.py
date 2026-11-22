"""Protocole du faux jeu n° 2 : exactement les requêtes réelles (v1 + v2a = 46), le v2b candidat (24) et un vivier
de sondes candidates pour la réserve (rsv_*, 256 tokens). Écrit sim/protocol2.jsonl. Aucun appel réseau."""
import json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import protocols as P  # noqa: E402

items = []
for l in open(os.path.join(ROOT, "data/raw/ep-01.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r["prompt_id"].startswith("v1_"):
        items.append((r["prompt_id"], r["repeat"], r["prompt"], 256, "v1"))
items += [(pid, rep, p, mt, "v2a") for pid, rep, p, mt in P.PROTOCOLS["v2a"]]
items += [(pid, rep, p, mt, "v2b") for pid, rep, p, mt in P.PROTOCOLS["v2b"]]

rng = random.Random(777)
RSV = {}
for k in range(6):
    mul = [(f"{rng.randint(101, 999)} * {rng.randint(101, 999)}", "") for _ in range(24)]
    RSV[f"rsv_pk_mul{k}"] = P._packed(mul, "Compute these products exactly.")
RSV.update({
    # style Alpaca (LoRA d'instructions)
    "rsv_al_fact": "Tell me an interesting fact about octopuses.",
    "rsv_al_story": "Write a short story about a robot who learns to paint.",
    "rsv_al_recipe": "Give me a recipe for a simple vegetable soup.",
    "rsv_al_tips": "Give three tips for staying healthy.",
    "rsv_al_explain": "Explain why the sky is blue.",
    "rsv_al_atom": "Describe the structure of an atom.",
    "rsv_al_poem": "Write a poem about autumn.",
    "rsv_al_class": "Classify the following animals as mammals, birds or reptiles and explain each choice.\n\nInput: dolphin, penguin, crocodile, bat, ostrich, gecko",
    "rsv_al_email": "Write a polite email asking a colleague to reschedule a meeting to next week.",
    "rsv_al_sum": "Summarize the main causes and consequences of the Industrial Revolution.",
    # domaines (LoRA spécialisé)
    "rsv_med1": "What are the typical symptoms, causes and first-line treatments of iron-deficiency anemia?",
    "rsv_med2": "Explain the mechanism of action of beta blockers and list their main side effects.",
    "rsv_law1": "Explain the difference between civil law and criminal law, with one example of each.",
    "rsv_law2": "What makes a contract legally binding? List and explain the essential elements.",
    "rsv_code1": "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring and two usage examples.",
    "rsv_code2": "Write a JavaScript function that removes duplicates from an array while preserving order, and explain how it works.",
    # histoires longues, autres langues
    "rsv_story_ru": "Напиши длинный, увлекательный рассказ (около 250 слов) о смотрителе маяка, который находит таинственное письмо.",
    "rsv_story_it": "Scrivi una storia lunga e fantasiosa (circa 250 parole) su una sarta che trova una chiave in un vecchio cappotto.",
    "rsv_story_pt": "Escreve uma história longa e imaginativa (cerca de 250 palavras) sobre um pescador que encontra uma garrafa com uma mensagem.",
    "rsv_story_nl": "Schrijf een lang, fantasierijk verhaal (ongeveer 250 woorden) over een vuurtorenwachter die een geheimzinnige brief vindt.",
    "rsv_story_pl": "Napisz długie, pomysłowe opowiadanie (około 250 słów) o latarniku, który znajduje tajemniczy list.",
    "rsv_story_tr": "Gizemli bir mektup bulan bir deniz feneri bekçisi hakkında uzun ve yaratıcı bir hikâye yaz (yaklaşık 250 kelime).",
    "rsv_story_ja": "不思議な手紙を見つけた灯台守についての、長くて想像力豊かな物語を書いてください（約400字）。",
    "rsv_story_zh": "请写一个长篇、富有想象力的故事（约300字），讲述一位灯塔看守人发现了一封神秘的信。",
    "rsv_story_de3": "Schreibe eine lange, fantasievolle Geschichte (etwa 250 Wörter) über einen Bäcker, der nachts ein seltsames Geräusch im Keller hört.",
    "rsv_story_fr3": "Écris une longue histoire imaginative (environ 250 mots) sur une horlogère qui répare une montre qui tourne à l'envers.",
    "rsv_story_es3": "Escribe una historia larga e imaginativa (unas 250 palabras) sobre un cartero que debe entregar una carta sin dirección.",
    # listes longues à forte entropie
    "rsv_words_fr": "Écris 80 mots français au hasard, séparés par des espaces. N'écris que les mots.",
    "rsv_words_de": "Schreibe 80 zufällige deutsche Wörter, durch Leerzeichen getrennt. Gib nur die Wörter aus.",
    "rsv_sent": "Write 15 unrelated random sentences, one per line, each about a different topic.",
    "rsv_nums2": "Write 100 random integers between 1000 and 9999, separated by spaces. Output only the numbers.",
    # continuations longues
    "rsv_cont_en": "Continue this text. Reply only with the continuation, no introduction.\n\nThe Danube is the second-longest river in Europe. It rises in the Black Forest of Germany and flows",
    "rsv_cont_fr": "Continue ce texte. Réponds uniquement par la suite du texte, sans introduction.\n\nLa Loire est le plus long fleuve de France. Elle prend sa source au mont Gerbier-de-Jonc, en Ardèche, et",
    "rsv_cont_code": "Complete the following Python code. Reply only with code, no explanation.\n\nclass LRUCache:\n    \"\"\"Least-recently-used cache with a fixed capacity.\"\"\"\n\n    def __init__(self, capacity):\n",
})
items += [(pid, 0, p, 256, "rsv") for pid, p in RSV.items()]
with open(os.path.join(ROOT, "sim/protocol2.jsonl"), "w", encoding="utf-8") as f:
    for pid, rep, p, mt, src in items:
        f.write(json.dumps({"prompt_id": pid, "repeat": rep, "prompt": p, "max_tokens": mt, "src": src}, ensure_ascii=False) + "\n")
import collections
print(len(items), collections.Counter(i[4] for i in items))
