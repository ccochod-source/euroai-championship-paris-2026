# Candidat Unlearning · correction par classe

Cette variante applique à chaque classe une intensité de correction légèrement
différente, puis divise les logits par 1,8. Elle conserve l’architecture et
tous les poids du ViT hors de la tête de classification. Les réglages viennent
du [sous-agent de simulation](https://app.devin.ai/sessions/63e22413987b4367bf5aa30643367f81).

Sur un échantillon local de 512 images d’oubli, ses trois scénarios valaient
0,0000, 0,1316 et 0,1341. Le même calcul sur le modèle record donnait
0,0000, 0,1090 et 0,0891. Sur 400 photos de Tiny ImageNet, les classes
prédites correspondaient à celles du modèle d’origine dans 74,3 % des cas
contre 53,0 % pour le record. La KL de contrôle monte de 1,54 à 1,81.
Ces mesures évaluent des données et des références locales. Tiny ImageNet
sert à mesurer l’accord entre modèles, avec ses étiquettes utilisées
seulement pour équilibrer l’échantillon.

## Reproduire le checkpoint

Extraire l’archive dans le dossier du kit officiel, à côté de `model.pt` et
`task_template.py`. Exécuter dans l’environnement Python du kit.

```bash
python reproduce_classwise.py model.pt head-4.0-changes.pt submission-classwise.pt
python validation.py submission-classwise.pt
```

Ou restaurer directement la tête retenue.

```bash
python checkpoint_changes.py restore model.pt head-classwise.pt submission-classwise.pt
```

La génération reproduit exactement les deux tenseurs du correctif testé.
Le checkpoint complet pèse 343 632 771 octets et a passé le chargement strict.

[Kit officiel](https://huggingface.co/datasets/SprintML/unlearning) ·
[Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet)
