# NUIT2 — consignes finales d'Orehn (01:40), valables jusqu'à son réveil
Objectif unique : AUGMENTER LES NOTES de façon robuste (classement final = 70 % caché), en autonomie complète dans les règles ci-dessous.

## Unlearning (priorité 1)
1. Suivre ABSOLUMENT le plan de Clément (BOARD.md / DEMANDES.md / SUBMISSIONS.md). Ses demandes passent en premier.
2. Quand son plan est épuisé ou qu'il n'a rien en file : continuer d'améliorer seul (UL_STRATEGIE.md, tes pistes ft6, profs out-of-fold).
3. Questions : si tu as besoin de contexte, écris dans BOARD.md une ligne « QUESTION -> Clément/session Devin : ... » et relis ses réponses à chaque boucle de 5 min.

## Model Equality (priorité 2, session simulateur)
Audit ROBUSTESSE_70 puis amélioration continue, sans dépenser la réserve ni soumettre (la réserve exige la confirmation tapée d'Orehn).

## Boucle d'apprentissage (les deux tâches)
- Après CHAQUE résultat (local ou officiel) : 3-5 lignes de post-mortem dans le journal de la tâche : ce qui a changé, pourquoi (hypothèse), si la prédiction était juste, test suivant.
- Toutes les ~2 h : lancer un sous-agent « critique » qui relit les résultats sans avoir produit le travail et cherche erreurs, fuites, surapprentissage du public, conclusions non prouvées. Intégrer ses remarques.
- Faire une veille ciblée (articles/code) quand une piste bloque.
- Anti-overfit : décider sur métriques locales + bootstrap ; écart public < 0,01 = bruit ; dans le doute « différent » (ME) ; acc locale >= 75 % (UL).

## Interdits inchangés
Pas de 2e job GPU, pas de sbatch/scancel sur les jobs des autres, pas de token affiché/copié, pas de dépense de réserve ME, pas de suppression de fichiers des autres ni de checkpoints soumis, SUBMIT_LOCK + 10 min.

## 07:00
Résumé clair en haut de results/status.md (ME) et results/unlearning_status.md (UL) : scores, ce qui a marché/raté, décisions à prendre par Orehn.
