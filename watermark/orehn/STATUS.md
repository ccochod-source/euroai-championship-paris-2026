# Watermark — session Orehn/simulateur (22/11, 10h46–10h56)

## Résultat : out/sub_orehn_a3.pt — mIoU hors échantillon (train, masques réels) 0.7619
Fichier : /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out/sub_orehn_a3.pt (1020 masques 128x128 bool, format vérifié). Rapport : out/sub_orehn_a3.json.
**Non soumis : je n ai pas de clé de soumission (pas de ~/.hack_api_key chez ansellem1). À soumettre par Bruno ou Orehn :**
  /p/scratch/training2666/dasilvalopes3/wm/submit.sh /p/scratch/training2666/cochod2/euroai-paris/Hackathon_Watermark_Localisation_Task/orehn/out/sub_orehn_a3.pt

## Ce qui a été fait (lecture seule des fichiers de Bruno)
1. Réassemblage avec les réseaux finis cette nuit (probabilités hors échantillon de wm/out) : pixelseal ftP6P (0.6419, seuil 0.35),
   trustmark ftT6P (0.7344, seuil 0.75), stegastamp moyenne ftP + ftS5P (0.8830, seuil 0.45). Global 0.7531. (assemble2.py)
2. Post-traitements morphologiques, seuils par régime, composantes, boîtes englobantes : gain nul en validation croisée, non appliqués. (assemble3.py)
3. Fusion avec la recherche de rectangles de Bruno (réseau OU rectangle quand acc >= 0.8, images non pleines) : (assemble4.py)
   trustmark 0.7344 -> 0.7546 (validation croisée 2 moitiés ; 83 images avec rectangle : 0.764 -> 0.806) : appliquée
   pixelseal 0.6419 -> 0.6472 (20 images avec rectangle : 0.819 -> 0.873) : appliquée
   stegastamp 0.8830 -> 0.8858 : sous le seuil de 0.004, non appliquée
   Global attendu 0.7619 (+0.009).

## Facteur limitant
PixelSeal (0.64 ; images pleines avec trous 0.60, partielles 0.68), puis TrustMark images pleines (0.70). StegaStamp est à 0.88.

## Réserve
Je ne sais pas quelle soumission donne le 0.75 public actuel ni son score local : si elle inclut déjà ces réseaux ET la fusion rectangles, le gain sera nul.

## Résultat public (11h00) : sub_orehn_a3.pt -> 0.7448, SOUS le meilleur de Bruno (0.7589, conservé)
- L estimation locale (0.7619 hors échantillon sur train) ne s est pas transférée. Écart avec le fichier de Bruno d hier 20h50 (sub_s3_v8.pt) : accord de 0.816 seulement entre les deux jeux de masques, donc deux solutions réellement différentes.
- Recherche d un ajout partant du fichier de Bruno (11h00–11h05) : le script qui a construit sub_s3_v8.pt n est pas sur le cluster (aucune référence à sub_s3 dans wm/s3 ; assemblage fait en local chez Bruno) et il n existe pas de masques TRAIN correspondants. Je ne peux donc valider localement aucune modification de son fichier (fusion avec mes réseaux, rectangles, vote). Sans validation, pas de 2e soumission : on s arrête là, la soumission retenue reste celle de Bruno.
- Leçons : (1) mon score hors échantillon venait de réseaux dont les seuils et les ensembles étaient choisis sur les mêmes 510 images, sans jeu de contrôle ; (2) la partie publique (~300 images) a un bruit d environ 0.01, l écart mesuré (−0.014) est faible mais va dans le mauvais sens ; (3) pour comparer deux solutions il aurait fallu les masques train du pipeline de Bruno.
