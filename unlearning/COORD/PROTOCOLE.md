# PROTOCOLE — coordination unlearning (Orehn, Clément = cochod2, Bruno = dasilvalopes3)

1. **Lire `BOARD.md` avant de lancer une piste.** Ne jamais prendre une piste déjà prise (statut « en cours » ou « fait »). Pour une variante proche : en parler à son propriétaire d'abord.
2. **S'inscrire dans `BOARD.md` avant de lancer** : qui, piste, statut, dossier, GPU/nœud.
3. **Toujours ajouter ses résultats** dans `RESULTS.csv` (une ligne par essai, même raté), avec les mêmes métriques locales (ci-dessous).
4. **Soumission** : jamais sans accord humain. Écrire le fichier `SUBMIT_LOCK` (qui, heure, checkpoint), l'annoncer dans `SUBMISSIONS.md`, attendre l'accord, soumettre, noter le score, supprimer `SUBMIT_LOCK`. Si `SUBMIT_LOCK` existe déjà : attendre. Délai de 10 min entre soumissions.
5. **GPU** : un seul nœud pour toute l'équipe, pas de `sbatch` personnel sans accord humain. Chacun n'écrit que dans son dossier (`orehn/`, `oof/`, `eugenia-trials/`, `output/bruno/`...) et dans `COORD/`.
6. Checkpoints : float32, même architecture, `strict=True`, < 450 Mo. Disque : supprimer les checkpoints intermédiaires inutiles (le worker GPU est tombé à 20:46 sur « Disk quota exceeded »).

## Métriques locales communes (colonnes de RESULTS.csv)
Étiquette y = argmax du modèle original sur l'image propre (il a tout mémorisé : confiance 0,9997).
- `f_acc`, `f_Lmean`, `f_Lstd` : accord avec y, moyenne et écart-type de la cross-entropy (T=1) sur les images à oublier propres. Original : 1,000 / 0,000 / 0,002.
- `alt_acc`, `alt_Lmean`, `alt_Lstd` : idem sur une copie altérée (x·0,8+0,1 en espace [-1,1], vue « bright »). Original : 0,846 / 0,61 / 1,71.
- `ctl_KL`, `ctl_agree` : KL(original || candidat) à T=2 et accord argmax sur des images de contrôle (préciser lesquelles dans `notes`).
- `rel_dw` : ||θ − θ0|| / ||θ0||.
- `score_officiel` : score du leaderboard si soumis, sinon vide.
