# European AI Championship Paris 2026 — équipe Eugenia

Hackathon CISPA / Fraunhofer AISEC à Epitech Paris, 21–22 novembre 2026. Ce dépôt rassemble le travail de l'équipe
Eugenia sur les trois pistes : code, scripts de lancement, plans, journaux d'essais et notes officielles.

Dernier classement public relevé (22/11, 10 h 33, avant les derniers envois) :

| Piste | Note publique | Dossier |
|---|---:|---|
| Watermark Localisation | 0,76 | [`watermark/`](watermark/) |
| Model Equality | 1,00 | [`model_equality/`](model_equality/) |
| Unlearning | 0,26 (record final 0,2642) | [`unlearning/`](unlearning/) |
| Global | 0,78, rang 10 sur 44 | |

Ces notes portent sur la partie publique des données. Le classement final est calculé sur une partie cachée et n'était
pas publié au moment de la création du dépôt.

## Ce que le dépôt ne contient pas

- les données du concours (images, archives `train.zip` / `test.zip`, `forget_images.npz`) ;
- les poids de modèles (`model.pt`, checkpoints de 340 Mo, modèles de watermark) ;
- les environnements Python et caches ;
- la clé d'équipe : les scripts de soumission contiennent `YOUR_TEAM_API_KEY`.

Les scripts ne sont donc pas exécutables tels quels sans les fichiers fournis par les organisateurs.

## Piste 1 — Unlearning (`31-unlearning`)

**Tâche.** Un ViT-B/16 à 100 classes et 3 500 images « à oublier ». Il faut rendre un modèle complet qui se comporte
sur ces images comme s'il ne les avait jamais vues, en gardant sa précision sur le reste (un modèle trop dégradé,
vers 70 %, est rejeté). Le serveur compare les images d'origine à des copies altérées cachées.

**Progression des notes officielles.**

| Étape | Méthode | Note |
|---|---|---:|
| 21/11 soir | Température globale, correction par classe, édition de la tête | 0,005 à 0,010 |
| 21/11 22 h | Fine-tuning complet vers des cibles « jamais vu » + KL vers le modèle d'origine sur des images de contrôle | 0,057 → 0,110 |
| 21/11 nuit | Cibles plus dispersées (`rpow` 2, 3, 4) | 0,163 → 0,202 → 0,209 |
| 22/11 01 h 39 | Idem, 6 derniers blocs seulement | 0,2101 |
| 22/11 02 h 21 | Cibles **par image hors-pli** (professeurs n'ayant pas vu l'image) | 0,2561 |
| 22/11 11 h 26 | Protection du reste (`gamma`) 4 → 3 | 0,2599 |
| 22/11 11 h 57 | Force des cibles 3 → 2,75 | **0,2642** |

**Ce qui a marché.** Remplacer une formule de cibles identique pour toutes les images par des cibles par image,
prédites par des professeurs entraînés sans cette image (out-of-fold). C'est l'essentiel du gain (0,21 → 0,26).

**Ce qui n'a pas marché.** Éditer seulement la tête ou une température ; pousser la force de l'oubli (force 4 : 0,2255) ;
plus de protection et un entraînement plus long (0,2514) ; abaisser en plus la vraie classe à `gamma` 3 (0,2471) ;
mélanger deux familles de professeurs (0,2366). Les mesures locales (perte, dispersion, précision estimée) n'ont pas
permis de prédire la note officielle : une extrapolation annonçait 0,27 à 0,30 pour un envoi noté 0,2347.

**Réglage du record.**

```bash
python -u orehn/src/finetune.py --name clem_oof_k275_g3_last6 \
    --targets orehn/cache/q_oof_k275.pt --gamma 3 --train last6 --lr 1e-4 --steps 3000
```

`q_oof_k275.pt` est `q_oof_clean` élevé à la puissance 2,75 puis renormalisé par image.

**Où regarder.**

- `unlearning/orehn/src/` : pipeline principal (`finetune.py`, `teacher.py`, `oof_vit.py`, `common.py`).
- `unlearning/orehn/logs/` : un JSON de mesures par essai.
- `unlearning/COORD/` : coordination à trois (`PROTOCOLE.md`, `BOARD.md`), toutes les soumissions avec leur note
  (`SUBMISSIONS.md`), tous les essais (`RESULTS.csv`), plans de nuit.
- `unlearning/codex/`, `unlearning/oof/`, `unlearning/eugenia-trials/`, `unlearning/devin-vm/` : pistes explorées
  avant le pipeline final (têtes, températures, distillation multi-vues, premiers professeurs hors-pli).

## Piste 2 — Watermark Localisation

**Tâche.** Pour chaque image 512×512 de test, rendre un masque binaire 128×128 des zones qui portent encore un
watermark invisible (PixelSeal, TrustMark ou StegaStamp, méthode et message connus). Note : mIoU moyen par image.

**Approche.** Deux voies, décrites dans `watermark/bruno-wm/s3/README.md` et `watermark/orehn/STATUS.md` :

- recherche de rectangles sans entraînement (« proposer et vérifier » avec le décodeur officiel) ;
- réseaux de segmentation ré-entraînés sur les vraies images d'entraînement, par méthode.

Constat important noté par l'équipe : pour TrustMark et PixelSeal, le watermark des images du concours n'est pas
corrélé pixel à pixel avec celui que produit l'encodeur fourni, donc les réseaux entraînés sur nos propres
watermarks échouent sur les vraies images.

Meilleure note publique consignée dans `watermark/orehn/STATUS.md` : 0,7589 (fichier de Bruno). Un assemblage estimé
à 0,7619 en local a obtenu 0,7448 en public.

**Où regarder.** `watermark/bruno/` et `watermark/bruno-wm/` (expériences, recherche de rectangles, segmentation),
`watermark/orehn/` (assemblages), `watermark/wmloc/` et `watermark/README.txt` (kit des organisateurs).

## Piste 3 — Model Equality (`33-model-equality`)

**Tâche.** 33 endpoints d'API servent des modèles de langage ; certains servent le même. Avec 100 réponses au plus
par endpoint, dire pour chaque paire si c'est la même famille et le même modèle (même checkpoint, quantification et
adaptateur LoRA ; un prompt système ou un décodage différent ne change pas le modèle). Note : MCC sur les paires,
10 % famille, 90 % modèle.

**Approche.** Un simulateur local (`model_equality/sim/`) reproduit le problème avec des modèles ouverts
(Llama-3.2-1B, Qwen2.5-1.5B/3B, SmolLM2-360M), leurs variantes quantifiées, LoRA et distillées, et des prompts système
de déguisement, pour concevoir et valider le protocole de requêtes et la règle de décision avant de dépenser le budget
réel. Note publique : 1,00.

**Où regarder.** `model_equality/sim/` (génération des jeux `make_spec*.py`, vraisemblances `lik*.py`, classifieur
de paires `pair_clf.py`, validations `validate_*.py`), `model_equality/task_template.py` (énoncé complet).

## Infrastructure

`infra/` : file d'attente partagée sur un nœud de 4 GPU du supercalculateur JURECA (`gpu_worker.py`,
`GPU_PARTAGE.md`), avec les scripts lancés et leurs journaux par GPU.

## Équipe

Clément, Orehn, Bruno, avec des agents de code (Devin, Claude, Codex) utilisés pour lancer et suivre les essais.
Les fichiers de coordination sont conservés tels quels, en français.
