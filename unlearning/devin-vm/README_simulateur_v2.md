# Simulateur Unlearning d’Eugenia

Le kit officiel fournit le ViT-B/16, les 3 500 images à oublier et le barème. Le
simulateur infère les étiquettes à partir du modèle d’origine. Il représente
les modèles réentraînés par trois hypothèses de prédictions et utilise des
compositions d’images pour mesurer la conservation des prédictions. Les
scores et taux d’accord locaux dépendent de ces hypothèses. Le serveur
mesure la précision réelle sur son jeu de contrôle.

## Reproduire la comparaison

Extraire l’archive dans le dossier du kit officiel, à côté de `model.pt`,
`task_template.py` et `submission.py`, puis activer l’environnement Python
du kit. L’archive contient les représentations mises en cache sur JURECA et
les corrections de la tête. Le calcul suivant fonctionne sur CPU.

```bash
export OMP_NUM_THREADS=4
python simulator_backtest.py
python selective_head.py
python compare_head_edits.py
```

- Le test rétrospectif retire chaque score officiel à tour de rôle avant
  d’ajuster l’échelle du simulateur. Sur les six essais précédant la
  sélection de la correction ancrée, l’erreur absolue moyenne était
  d’environ 0,00077 dans le troisième scénario et 0,00270 avec la médiane.
  Les six résultats ont servi à comparer les scénarios.
- La correction ancrée part du modèle record et cherche à conserver les
  prédictions d’origine sur des compositions d’images. Elle laisse de côté
  trois échantillons de validation de 192, 256 et 512 images. Avant l’envoi,
  elle améliorait le troisième scénario sur les trois échantillons. Le
  deuxième scénario la classait derrière le record sur les trois échantillons.
- La comparaison des variantes de la tête sépare la mise à l’échelle globale
  des logits de la correction ciblée. Elle produit des diagnostics sur les
  mêmes trois échantillons.

## Reconstituer un checkpoint

Les correctifs internes contiennent les tenseurs `head.weight` et
`head.bias`. Pour produire un checkpoint complet à partir d’un correctif
direct, les fusionner avec l’état du modèle officiel.

```python
import torch

original = torch.load("model.pt", weights_only=True, map_location="cpu")
head = torch.load("loop/anchored-v2/anchored-head-a0.02-r0.0001-1.pt",
                  weights_only=True, map_location="cpu")
original.update(head)
torch.save(original, "submission-anchored.pt")
```

```bash
python validation.py submission-anchored.pt
```
