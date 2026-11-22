"""Score attendu (0.1 x MCC famille + 0.9 x MCC modèle, sur toutes les paires) de plusieurs soumissions sous plusieurs vérités.
    python sim/mcc_scenarios.py"""
import itertools, math
FAM = {"L": "05 09 10 11 13 15 16 19 27 29 31", "A": "01 02 04 12 14 17 18 23 24 25 33", "B": "03 06 07 08 20 21 22 26 28 30 32"}
fam = {e: f for f, s in FAM.items() for e in s.split()}
def part(S):   # S : {famille: ensemble d'endpoints dans S} ; le reste en singletons
    return {e: (f + "_S" if e in S[f] else e) for e, f in fam.items()}
def mcc(t, g):
    tp = sum(a and b for a, b in zip(t, g)); tn = sum((not a) and (not b) for a, b in zip(t, g))
    fp = sum((not a) and b for a, b in zip(t, g)); fn = sum(a and (not b) for a, b in zip(t, g))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
def score(truth, pred):
    P = [(a, b) for a, b in itertools.combinations(sorted(fam), 2) if fam[a] == fam[b]]
    return 0.1 * 1.0 + 0.9 * mcc([truth[a] == truth[b] for a, b in P], [pred[a] == pred[b] for a, b in P])
s = lambda x: set(x.split())
FINAL = {"L": s("10 13 16 29 31"), "A": s("02 04 23 33"), "B": s("06 08 20 30 32")}
def plus(base, **kw):
    out = {f: set(v) for f, v in base.items()}
    for f, x in kw.items():
        out[f] |= s(x)
    return out
def minus(base, **kw):
    out = {f: set(v) for f, v in base.items()}
    for f, x in kw.items():
        out[f] -= s(x)
    return out
SUBS = {"finale actuelle (v6 + ep-13)": FINAL, "+ ep-12": plus(FINAL, A="12"), "+ tronqués 27/14/07": plus(FINAL, L="27", A="14", B="07"),
        "+ ep-12 + tronqués": plus(FINAL, L="27", A="12 14", B="07"), "+ ep-12 + tronqués - ep-02": minus(plus(FINAL, L="27", A="12 14", B="07"), A="02")}
T5 = plus(FINAL, A="12")
TRUTHS = {
 "V1 : 5 S par famille, plain QA = ep-12, tronqués = variantes": T5,
 "V2 : 6 S par famille (tronqués dans S), plain QA = ep-12": plus(T5, L="27", A="14", B="07"),
 "V3 : comme V1 mais plain QA = ep-17 (ep-12 variante)": plus(FINAL, A="17"),
 "V4 : comme V2 mais plain QA = ep-17": plus(FINAL, L="27", A="14 17", B="07"),
 "V5 : comme V2 mais ep-02 n'est pas S": minus(plus(T5, L="27", A="14", B="07"), A="02"),
 "V6 : comme V2 mais ep-13 n'est pas le plain (plain Llama = ep-19)": plus(minus(plus(T5, L="27", A="14", B="07"), L="13"), L="19"),
 "V7 : comme V2 mais ep-32 n'est pas le plain (plain QB = ep-26)": plus(minus(plus(T5, L="27", A="14", B="07"), B="32"), B="26"),
 "V8 : comme V1 mais les 3 personas (16, 23, 20) ne sont pas S": minus(T5, L="16", A="23", B="20"),
}
names = list(SUBS)
print("vérité".ljust(72) + " | " + " | ".join(n[:22].rjust(22) for n in names))
for tn, T in TRUTHS.items():
    print(tn.ljust(72) + " | " + " | ".join(f"{score(part(T), part(S)):22.3f}" for S in SUBS.values()))
