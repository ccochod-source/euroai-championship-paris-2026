# DEMANDES — essais que Clément (ou sa session) demande à la session d Orehn de lancer

Fichier créé le 22/11 à 01h30 par la session d Orehn (il n existait pas) : Clément, corrige les règles ci-dessous si elles ne te conviennent pas.

## Règles
1. Une demande = une ligne du tableau, statut **NOUVELLE**. Seule la session d Orehn change le statut : NOUVELLE → PRISE → EN COURS → TERMINEE (ou REFUSEE, avec la raison).
2. La commande doit être un appel à un script existant du dossier `unlearning/` (ex. `orehn/src/finetune.py --name ... --A ...`), écrire uniquement dans `unlearning/orehn/ckpt`, `unlearning/output` ou ton dossier, et durer moins de 40 min. Pas de sbatch, srun, scancel, pas de suppression de fichiers d autrui, pas de token.
3. Lancement via `gpu_queue/gpuN/` avec le worker existant (un seul job GPU pour l équipe). GPU 3 par défaut ; GPU 0 seulement s il est libre côté Model Equality.
4. La session d Orehn lit ce fichier toutes les 5 min, donne la priorité à tes demandes quand le GPU est disputé (en alternant avec ses propres essais), puis remplit log + résultats ici et dans RESULTS.csv.
5. Aucune soumission au leaderboard à partir de ce fichier : il faut une ligne « demandé » dans SUBMISSIONS.md et le SUBMIT_LOCK.

## Demandes
| N° | Heure | Demandeur | Commande (arguments de finetune.py ou script) | GPU souhaité | Statut | Log | Résultats (CE moy / é.-t., acc, retain_est) |
|---|---|---|---|---|---|---|---|
