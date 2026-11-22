# Eugenia | SprintML Unlearning

Les scripts tournent avec le kit officiel Unlearning dans
`/p/scratch/training2666/cochod2/euroai-paris/unlearning/eugenia-trials`.
Ce dossier pointe vers le modèle, les images, l’environnement Python et le
fichier `task_template.py` du dossier parent. Le modèle de départ et le kit
officiel restent disponibles auprès de [SprintML](https://huggingface.co/datasets/SprintML/unlearning).

## Reproduire un essai sur JURECA

```bash
jutil env activate -p training2666
cd /p/scratch/training2666/cochod2/euroai-paris/unlearning/eugenia-trials
sbatch --job-name=head run-candidate.sh head 1.0 --fit-all
```

Slurm utilise un GPU A100, la partition `dc-gpu` et la réservation
`cispahack`. Chaque job écrit un checkpoint et un journal avec son identifiant
dans `output/`. Le journal contient la configuration, la marge sur la
classe alternative la plus probable et des statistiques de confiance, de
perte et de KL.

Pour lancer jusqu’à quatre essais en parallèle sur le même nœud, place
`submit-candidates.sh` et `run-candidate.sh` dans le dossier partagé, puis
lance par exemple :

```bash
bash submit-candidates.sh head 1.0 gradient 1.0 influence 1.0 view_ridge 1.0
```

Le script vérifie qu’aucun membre d’Eugenia n’a de job en cours ou en attente
sur `cispahack`. Chaque essai utilise un GPU de l’allocation et écrit son
propre journal et checkpoint dans `output/`.

Les essais sans `--fit-all` excluent 192 images de l’ajustement. Les
mesures locales portent sur ces images, leurs retournements et leurs
recadrages. Les essais avec `--fit-all` entraînent sur les 3 500 images. Le
champ `heldout` de ces journaux désigne le même échantillon de 192 images,
inclus dans l’ajustement dans ce cas. L’accord mesuré est relatif à la
classe prédite par le modèle initial. La précision de contrôle et le score
de référence sont évalués par le serveur.

## Vérifier un checkpoint avant l’envoi

```bash
.venv/bin/python validation.py output/head-4.0-15901766.pt
```

Le validateur vérifie les tenseurs flottants finis, la limite de
450 000 000 octets et le chargement strict par le ViT-B/16 du kit.
Le serveur applique un seuil de 70 % sur un jeu de validation distinct du
jeu de contrôle et impose un délai entre les soumissions.

## Simuler et comparer les réglages

Place les scripts de l’archive dans un dossier avec `task_template.py`,
`model.pt` et `forget_images.npz` du kit. Conserve l’emplacement
`output/head-4.0-15901766-changes.pt` de la correction fournie.

```bash
OMP_NUM_THREADS=6 .venv/bin/python simulate_score.py
```

Le premier passage extrait les représentations de 192 images et de leurs
variantes, puis les place dans `loop/simulator-features.pt`. Les passages
suivants réutilisent le cache et comparent 18 réglages. Le rapport
`loop/simulator.json` contient les résultats par réglage et par hypothèse.

Le simulateur applique la structure du barème avec un poids de 50 % pour les
originaux et de 50 % pour les variantes, la KL à température 2, et les
statistiques de perte moyenne et d’écart-type. Trois ensembles de prédictions
issus de vues recadrées et de plusieurs niveaux de calibration servent de
référence locale. Il mesure la conservation des prédictions sur des
compositions d’images synthétiques. Les labels d’oubli sont approximés par
les réponses du modèle original. La précision de contrôle et le seuil de
validation restent mesurés par le serveur.
