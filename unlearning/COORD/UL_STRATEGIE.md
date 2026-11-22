# UNLEARNING — PLAN POUR DÉPASSER 0.12 (synthèse recherche, 22/11 ~00:00, Claude/Orehn)
Diagnostic: un seul A global pour toutes les images => impossible de reproduire (1) le STD de la loss du gold
(gold = modèle qui n'a jamais vu l'image: la plupart des images restent faciles/bonne classe, une queue d'images
dures a une loss élevée) et (2) QUELLE mauvaise classe le gold prédit par image (prediction_score, KL T=2).

## Checks rapides (10 min)
1. Histogramme des pseudo-labels (argmax modèle original sur forget). Classes sur-représentées (>>35/classe)
   => le gold oublie en partie ces classes: baisser leur masse séparément.
2. Init publique ? cosinus patch_embed/block0 vs timm deit3_base_patch16_224 / vit_base_patch16_224 (augreg).
   Si match => le gold a été fine-tuné depuis là => "retrain" bon marché possible depuis la même init.

## Stratégies classées
1. PROFS OUT-OF-FOLD (priorité 1). K=5 folds des 3500 images. Pour le fold k: entraîner sur les 4 autres folds
   (pseudo-labels, label smoothing 0.1), prédire le fold k => logits z_oof "jamais vu".
   - version rapide: DINOv2-B/14 ou CLIP-L/14 gelé + régression logistique (minutes).
   - version forte: init publique (si trouvée) ou réinit blocs 10-11 + head; AdamW lr 1e-5..3e-5 blocs, 1e-4 head, 5-15 epochs.
   - moyenner 2-3 seeds/splits (le gold est une moyenne de modèles).
   - CALIBRER: cible z_t = beta*z_oof/T + (1-beta)*(z_orig avec top-1 baissé). Choisir T,beta pour que
     moyenne ET std de la CE (pseudo-labels) sur forget = stats "unseen" cibles.
2. A PAR IMAGE (fallback): difficulté d_i = rang moyen de (kNN DINOv2 leave-one-out: fraction de voisins
   d'une autre classe) + (chute de confiance sous augmentation forte/bruit) + (proba OOF du pseudo-label).
   A_i = A_max*sigmoid((d_i-tau)/w). Masse libérée -> rangs 2-5 de l'original proportionnellement à leur softmax
   (PAS uniforme). Effet: std de loss bimodale, beaucoup moins de perte d'accuracy qu'un A=0.05 global.
3. TERME DE RÉTENTION sur données substituts: L = L_forget + lambda*KL(T=2)/MSE logits centrés vs original
   sur images NON-forget (ImageNet val/COCO/images générées), lambda 0.5-2. Vise directement retain_score
   (KL(p_orig||p_sub) sur contrôle) => permet d'oublier plus fort sans passer sous 70%.
   (Précédent NeurIPS 2023: Amnesiacs, 8e place "Forget" — MSE > CE.)
4. COPIES ALTÉRÉES = 50% du forget score: entraîner sur K vues augmentées par image (RandomResizedCrop 0.5-1,
   ColorJitter, blur, JPEG 30-70, flip, petite rotation); avec #1 calculer la cible par vue.
5. Matcher les LOGITS (MSE logits centrés) + petit KL T=1 et T=2. En cas de doute sur la mauvaise classe:
   étaler sur top-3/5 de l'original (KL(gold||sub) punit p_sub~0).
6. Paramètres: blocs 8-11 + LayerScale + head (option masque SalUn top 10-50% |grad|). AdamW lr 5e-6..3e-5,
   3-10 epochs, cosine, checkpoint par epoch; choisir le checkpoint dont la loss moyenne forget ~ cible unseen (SCRUB+R).
   Alternative LoTUS (CVPR'25): température auto exp(2*(Acc_forget-Acc_unseen)), ViT lr 1e-6, 3 epochs.
7. Reset/bruit des 1-2 derniers blocs + réparation par distillation sur substituts (seulement si substituts bons).
8. À ÉVITER: Bad Teacher, UNSIR/bruit max-erreur, relabel aléatoire, NegGrad+, SSD (poussent vers uniforme/aléatoire,
   l'inverse de la structure du gold).

## DICHOTOMIE SANS OVERFIT
- 1 seul bouton s (A_max, ou beta/T). Proxies LOCAUX indépendants du prof d'entraînement:
  mean_std proxy (CE moy/std forget+vues vs stats unseen), KL T=2 vs un AUTRE prof (DINOv2 si on a entraîné sur OOF finetuné, ou l'inverse),
  retain proxy (acc ratio + KL sur substituts). GARDE: acc estimée >= 75%.
- Bissection sur s jusqu'à loss moyenne = cible, puis golden-section sur log s dans [s*/2, 2s*].
- Bruit: bootstrap des proxies sur sous-ensembles aléatoires de 30% => écart < bruit = pas un signal.
- Soumettre 3 modèles max: s*, 0.7s*, 1.4s*. Final = meilleur proxy parmi ceux dans le bruit du meilleur LB.
- Garder la pipeline A-only comme fallback. Respecter SUBMIT_LOCK + 10 min.

Sources: arxiv 2406.09073 (NeurIPS 2023 MU challenge), SCRUB 2302.09880, SalUn 2310.12508, LoTUS 2503.18314,
JiT 2402.01401, RUM 2406.01257, junhyunb.github.io/en/projects/neurips2023-unlearning/
