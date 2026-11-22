# Clément (+ Devin) — état Unlearning, 21/11 21:00

Complément à BOARD.md / RESULTS.csv / SUBMISSIONS.md (mis à jour 21h05).
À recopier dans BOARD.md / RESULTS.csv / SUBMISSIONS.md quand ils existent.

## Accès
- Le compte cochod2 n'a PLUS de budget GPU sur JURECA depuis 20:38 (mail JSC). Aucun job possible sous cochod2.
- Fichiers et connexion toujours accessibles.

## Soumissions officielles (score public)
| Heure | Candidat | Score |
|---|---|---|
| - | modèle d'origine | 0.0 |
| - | tête corrigée x1.0 / x1.07 / x1.10 | 0.0082 / 0.0099 / 0.0101 (record) |
| - | température globale T=3.789 | 0.0053 |
| - | seconde tête x1.035 | 0.0059 |
| 18:49 | correction par classe | 0.0084 |
| 20:47 | 6 derniers blocs distillés vers professeur DINOv2 hors-pli (oof-d6-lr3e-5-s300) | 0.0076 |

Prochaine soumission possible côté délai : dès maintenant (dernier envoi 20:47). Aucune soumission prévue sans accord.

## Ce que ça suggère (3 points, pas une certitude)
- Perte moyenne sur forget : record = 1.58 -> 0.0101 ; candidats à perte 0.33 -> 0.0053 et 0.0076.
- Les modèles gold semblent beaucoup moins confiants sur forget que le professeur DINOv2 (perte 0.44, 85.7 % d'accord avec le ViT d'origine).
- Tiny ImageNet est un mauvais contrôle de conservation (images du concours = images générées).

## Pistes déjà faites (ne pas refaire)
- Corrections de tête (échelle, par classe, SALSA, marges, ridge, ancrage public) : plafond ~0.010.
- JiT, LF-SSD : rejetés en local.
- Dernier bloc MLP sur activations en cache : gain local marginal.
- Distillation blocs vers professeur hors-pli, tau=1 : 0.0076 officiel.

## Piste que je prends
- Même distillation de blocs avec professeur adouci (tau = 3, 5, 8 -> perte visée 1.8 à 3.2).
- Prêt à lancer, mais il faut le GPU d'un coéquipier :
  CUDA_VISIBLE_DEVICES=<n> bash /p/scratch/training2666/cochod2/euroai-paris/unlearning/oof/run-sweep.sh /p/scratch/training2666/cochod2/euroai-paris/unlearning/output/bruno 6:3e-5:3 6:3e-5:5 12:3e-5:3
  (format profondeur:lr:tau, ~3 min par réglage, écrit journaux + correctifs dans output/bruno, ne soumet rien)

## Fichiers
- Scripts : unlearning/oof/ (oof_finetune.py, run-sweep.sh, professeur oof-wd0.01.pt)
- Résultats du job 15902873 : unlearning/output/oof-15902873/ (journaux + correctifs par pas)

