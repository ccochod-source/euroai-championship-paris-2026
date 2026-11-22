# Résultats Codex — Unlearning

## Premier lot, 21/11 vers 22h20 Paris

Gold inaccessible. Les accords ci-dessous sont avec le pseudo-label original, **pas des accuracies de validation**. Forget : 3 500 images complètes ; contrôle : 2 000 images IA publiques tenues à part, mêmes indices que chez Orehn.

| Essai | CE propre moyenne / std | CE bright moyenne / std | KL contrôle T2 | Accord contrôle | Décision locale |
|---|---:|---:|---:|---:|---|
| Original reproduit | 0,000322 / 0,002430 | 0,61364 / 1,70889 | ~0 | 1,000 | référence concordante |
| NPO initial, last4, 700 pas | 9,887 / 9,803 | 22,256 / 8,693 | 0,1278 | 0,8440 | rejeté : forte dérive, surtout sur vues altérées |
| Professeur retain-only blocs 6/7/8, last4, 700 pas | 0,828 / 0,987 | 1,883 / 1,580 | 0,0745 | 0,8705 | utile comme nouvelle direction, pas encore recommandé en soumission |

Les deux checkpoints sont en float32, sans NaN/Inf, ~343,6 Mo, rechargés depuis disque en `strict=True`. Aucune soumission faite par Codex. Une première exécution de référence a échoué avant entraînement (scheduler au pas zéro), corrigée et consignée dans RESULTS.csv.

Le professeur retain-only utilise les 100 classes et n'apprend rien à partir des pseudo-labels forget. Son accord de calibration avec l'original sur proxies est de 45,3%, 53,3%, 59,1% pour les blocs 6, 7, 8 ; la moyenne de leurs prédictions sur forget implique une CE de 2,50 / std 2,21 et un accord de 35,7%. Ce n'est pas un gold : le backbone reste contaminé et les images publiques peuvent être hors domaine.

Le record officiel d'équipe a changé pendant le travail : **0,0572**, obtenu par Orehn sur `ft2_A045` (source : COORD/RESULTS.csv et SUBMISSIONS.md). Cela rend obsolète le constat d'un plafond strict à 0,01. Le diagnostic des erreurs de prédiction et des vues altérées reste pertinent.

## Suite préparée

- NPO corrigé : pénalité de restauration au-dessus du plafond, KL conditionnelle préservant les classes alternatives, poussée négative seulement quand l'original est confiant, exposition explicite à diverses luminosités/contrastes.
- Professeur retain-only en ligne : recalcule les cibles sur chaque vue augmentée, au lieu d'imposer la même distribution à toutes les vues. Fine-tuning complet pour tester le sous-apprentissage observé au premier lot.
- Évaluer le record Orehn avec notre protocole complet pour une comparaison identique. Aucune modification de son checkpoint.

L'audit `logs/sensitivity.json` applique la formule à 18 scénarios **synthétiques** de gold. Le classement change selon les hypothèses. Ces nombres ne prédisent aucun score officiel et ne servent pas à choisir une soumission.

Diagnostic de formule et limites : DIAGNOSTIC.md. Reproduction du premier lot : `bash codex/run_initial.sh` via une file GPU autorisée uniquement. Suite : `run_followup.sh`, à déposer uniquement lorsqu'un GPU 2 ou 3 est libre. Ne pas relancer ces scripts sur le login.

Analyse semantique : le professeur retain-only predit 99 classes distinctes sur forget ; masse moyenne sur les 55 classes absentes de forget : 36,99%, soit 43,88% conditionnellement aux classes alternatives. La classe alternative principale coincide avec celle des perturbations de l original dans 29,49% des cas : direction differente, pas de preuve que ce soit le gold.
