# Diagnostic Unlearning — Codex, 21 novembre 2026

Source de vérité : `task_template.py`, documentation de notation fournie avec le modèle. Gold et labels de validation non disponibles.

Pour chaque groupe g (propre, altéré),

`M_g = max(0, 1 - (|mu - mu_gold| + |sigma - sigma_gold|) / d_original)`

`P_g = max(0, 1 - KL(gold_T2 || candidat_T2) / KL(gold_T2 || original_T2))`

Puis `forget = mean_g(M_g) * mean_g(P_g)` et `score = forget * retain`, avec `retain = clip(acc/acc_orig,0,1)/(1+KL(original_T2 || candidat_T2))`. Sous 70% en validation : zéro.

Attention : c'est le **produit des moyennes**, pas la moyenne des produits. Les KL sont directionnelles. La CE et ses moments sont à T=1 ; les distributions de prédiction sont à T=2. Le code officiel documente ces conventions mais ne fournit pas le calcul serveur ni ses dénominateurs gold.

## Ce que les données établissent

- Original : accord 100%, CE proche de zéro. Les copies bright ont déjà CE 0,61 / std 1,71. On ne peut pas appliquer le même degré d'oubli sans mesurer les deux groupes.
- Tête record : CE 1,58 / std 0,46 ; une moyenne élevée avec peu de dispersion. Elle ne démontre pas que le gold a une moyenne élevée, ni que le retain est bon (contrôle Tiny ImageNet hors domaine).
- Fine-tuning A07 : CE 0,079 / std 0,260, score officiel 0,0059. Cela ne permet pas d'isoler le manque de force, les mauvaises classes alternatives ou le retain.
- **Signal spécifique aux copies altérées** : A07 a abaissé la CE bright de 0,61 / std 1,71 à 0,391 / std 0,930, avec un accord passant de 0,846 à 0,877. Il oublie un peu les images propres tout en devenant plus confiant sur bright. Si le gold est moins performant sur ces copies que l'original, cette branche part dans la mauvaise direction. La cible statique par image peut masquer cette asymétrie. Le sens réel reste conditionné au gold inconnu.
- Les sondes CV entraînées sur forget utilisent les autres membres du groupe oublié comme données d'apprentissage. Elles ne simulent donc pas nécessairement un gold auquel tout ce sous-groupe manque. Elles ne couvrent en outre que 45 classes.
- Les sondes des derniers blocs peuvent exploiter la mémorisation déjà présente dans le backbone, même si leur tête est hors pli. Une sonde CV ne garantit pas un professeur « jamais vu ».
- Les images IA publiques sont un contrôle sans labels de vérité. Accord avec original et KL n'établissent PAS l'accuracy de validation, ni le passage du seuil 70%.

## Pourquoi 0,01 est plausible sans bug d'échelle

Exemple illustratif, PAS une estimation de nos facteurs : 0,12 de gain sur les moments × 0,12 de gain sur les prédictions × 0,70 de retain = 0,0101. Pour 0,32 avec retain 0,8, il faut un produit moments×prédictions de 0,4, donc environ 0,63 pour chacun s'ils sont équilibrés. Des ajustements globaux ne suffisent pas à reproduire les erreurs spécifiques et leur variabilité.

Un candidat peut perdre des points en oubliant trop : si sa distance au gold dépasse celle de l'original, le facteur correspondant est tronqué à zéro. Les gains ne s'additionnent pas ; on doit réduire les deux distances gold, sur les deux groupes, sans dériver sur le retain.

Pour le groupe propre et un gold dont moyenne/std dépassent celles du candidat, `M ≈ (mu_candidat + sigma_candidat)/(mu_gold + sigma_gold)` puisque les moments originaux sont presque nuls. Cela explique la faible progression d'un candidat sous-appris. Cette approximation n'est PAS valable pour bright : l'original y a déjà une grande variance, et le dénominateur peut être bien plus petit. Un changement modeste sur bright peut donc être pénalisé fortement.

Autre prudence : la distribution gold utilisée en KL est une moyenne de plusieurs modèles à T=2. On ne peut pas en déduire en général leurs moments de CE à T=1 en mettant simplement cette moyenne au carré. Les moments « impliqués » par nos cibles sont une aide de réglage d'un modèle déterministe, pas une reconstruction des statistiques gold.

## Deux expériences nouvelles, distinctes des pistes réservées

1. **Professeur appris sur retain seulement, avant les blocs mémorisants.** Sondes aux blocs 6/7/8, CLS + moyenne des patches. Apprentissage sur 12k images publiques avec distributions de l'original sur 100 classes ; hyperparamètres sur les 2k suivantes ; 2k finales exclues de l'entraînement et de la calibration. Aucun label forget pour entraîner/sélectionner le professeur. Distillation sur forget augmenté. Risques : proxies hors domaine, backbone toujours contaminé ; ce n'est pas un gold.
2. **Préférence négative bornée sans professeur forget.** Réduire la vraisemblance des pseudo-labels forget relativement à l'original, avec saturation et cap CE par image ; préserver KL original sur de vraies images publiques séparées. Tester si l'hypothèse des cibles par image est le verrou. Objectif adapté de [Negative Preference Optimization, Zhang et al., COLM 2024](https://arxiv.org/abs/2404.05868), initialement étudié pour les LLM : son transfert au ViT est une expérience, pas un résultat établi. Risque : les classes alternatives émergentes peuvent être mauvaises ; aucune garantie de score.

Le script de train est propre à Codex, lit seulement les données/helpers existants et écrit seulement dans codex/ et en ajout de RESULTS.csv. Vérification sauvegarde : float32, finitude, taille <450 Mo et rechargement strict depuis disque. Pas de soumission.

## Décisions fondées sur les mesures

Conserver logits propres, bright, flip, crop, noise, blur et contrôle pour chaque essai. Rejeter comme candidats pratiques les dérives retain manifestes ; ne jamais appeler leur accord proxy « accuracy ». Comparer la diversité des pertes et les classes alternatives ; le classement local ne vaut pas classement gold. Sans facteurs serveur, la cause exacte et le niveau gold restent non identifiables. Orehn seul décide d'éventuelles soumissions.

## Diagnostic complementaire — 22/11 01:26

Forget contient 45 classes sur100, avec exactement130 images dans les classes [0, 7, 8, 29, 30, 31, 33, 38], et129 dans la classe6. Ce nest pas un echantillon aleatoire equilibre du controle public. Cela ne prouve pas que ces classes sont entierement absentes du retain (effectifs dentrainement inconnus).

Un professeur OOF entraine sur4/5 de forget conserve encore des exemples de ces classes. Il ne simule donc pas necessairement un gold entraine sans TOUT forget. Son ancienne CE0.440/std0.615, acc85.7%, ne peut pas etre adoptee aveuglement comme les moments gold. Le nouvel OOF sert a obtenir une structure de predictions hors pli ; les moments sont calibres sur des probes retain-only distinctes, avec evaluation reservee.

Le score multiplie deux facteurs clips, chacun normalise par lecart original-gold : obtenir la bonne moyenne sans la dispersion ET les bonnes alternatives laisse un facteur proche de zero. Les copies alterees pesent autant que les images propres. Lamelioration officielle jusqua0.2019 accompagne la dispersion (rpow), sans demontrer que toute hausse de std est bonne. Le cap exige une selection locale multivue, pas une extrapolation du leaderboard.

Limite majeure : aucune estimation locale de gold nest certifiee. Le prof evaluation utilise le backbone original et peut avoir une erreur correlee avec les probes de calibration ; il est distinct du DINO dentrainement et sa tete nest jamais utilisee pour regler les cibles. Le test reserve decalage/rotation/bruit et les sous-echantillons30% limitent certains biais, pas ce biais de domaine.

OOF doit porter sur le PROFESSEUR, pas sur les images exposees au fine-tuning final : oof/oof_finetune.py ancien excluait700images du fine-tuning, qui restaient alors memorisees. Le nouveau FT voit les3500images, chaque cible etant produite par des tetes lineaires qui ont exclu cette image et toutes ses vues de leur apprentissage. Le controle hors entrainement est constitue dimages non-forget et de transformations reservees, pas dimages forget laissees intactes.
