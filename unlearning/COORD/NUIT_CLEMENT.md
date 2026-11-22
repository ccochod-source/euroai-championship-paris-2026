# Plan de nuit Unlearning — Clément (session Devin) -> agent d'Orehn
Version 3, 22/11 ~01h50. Remplace les versions 1 et 2. Clément indique qu'Orehn a demandé à son agent de suivre ce plan. PROTOCOLE.md prime en cas de conflit. Ma session n'a plus accès à JURECA jusqu'au réveil de Clément : je ne lance, ne lis et ne soumets rien.

## Constat (scores officiels, public split)
| checkpoint | réglage | std clean / crop | retain_est | score |
|---|---|---|---|---|
| clem_A03r2_g4 | train all, rpow 2 | 2,59 | 0,757 | 0,1631 |
| clem_A03r3_g4 | train all, rpow 3 | 3,26 | 0,750 | 0,2019 |
| clem_A03r4_g4 | train all, rpow 4 | 3,66 / 3,11 | 0,750 | 0,2091 |
| ft6_A03r5_g4_last6_t1 | last6, rpow 5, t1 1.0 | 3,76 / 3,34 | 0,752 | **0,2101** (record) |
(tous : A=0,3, alpha 4, gamma 4, lr 1e-4, 3000 pas)
La famille « formule make_targets + rpow » PLAFONNE vers 0,21 : +0,007 puis +0,001, que l'on entraîne tout ou last6, avec ou sans t1, et même quand la std crop monte. Ne plus dépenser de GPU ni d'envoi sur rpow, k de last<k>, t1, gamma ou lr dans cette famille.
Déjà écarté (non soumis) : lr > 1e-4 (retain_est < 0,70) ; A=0,15 rpow 2 (0,741) ; gamma 6/8 + t1 (0,740/0,736).

## Étape 1 — cibles par image out-of-fold (PRIORITÉ ABSOLUE, passe devant tout dans gpu3)
Finir l'extraction + les 15 professeurs OOF (Codex), produire le fichier de cibles par image.
Premier essai = UN SEUL changement par rapport à un point noté : réglage de clem_A03r4_g4 (train all, gamma 4, lr 1e-4, 3000 pas) avec --targets <fichier OOF>. Pas last3, pas d'autre perte en même temps, sinon on ne saura pas ce qui a joué.
Contrôle avant d'entraîner : sur les cibles elles-mêmes, noter l'accuracy vs étiquette, la CE moyenne et sa std (les comparer à 0,31 / 4,10 / 3,66 du record). Écrire ces trois chiffres dans BOARD.md.

## Étape 2 — dichotomie sur la force des cibles OOF
Trois forces (les 1 / 0,7 / 1,4 prévues par Codex). Garder celles à retain_est >= 0,75, soumettre la meilleure, puis couper en deux l'intervalle entre les deux meilleures notes. Une seule variable.

## Étape 3 — contrôle de forme (déjà en file, gpu0)
c3_clement_A045r4_g4 (A 0,45, rpow 4, train all) : même dispersion visée mais plus d'images bien classées. Lecture locale seulement ; ne le soumettre que s'il reste un envoi et que l'étape 1 n'a rien donné. S'il était noté >= 0,21, c'est la forme et non la force de l'oubli qui compte : viser alors des cibles OOF plus douces à l'étape 2.

## Conditions obligatoires pour tout envoi
- retain_est >= 0,75 ;
- std tr/crop et tr/noise.2 >= 85 % de la std clean ;
- validation contre model.pt (clés, formes, float32, fini, < 450 Mo) ;
- SUBMIT_LOCK, 10 min d'intervalle, ligne dans SUBMISSIONS.md et RESULTS.csv ;
- AU PLUS 3 envois cette nuit, chacun décidé par Orehn / son agent. Un candidat de la famille rpow ne vaut plus un envoi.

## Robustesse
- Limite du NOMBRE de fichiers du scratch atteinte (annonce organisateurs 01h33) : une écriture peut échouer. Si un essai plante à la sauvegarde : relancer une fois, puis passer au suivant. Ne pas créer de nouveaux venv ni de caches en milliers de petits fichiers ; préférer un seul .pt / .npz.
- Ne pas supprimer : orehn/ckpt/ft6_A03r5_g4_last6_t1.pt (record), clem_A03r4_g4.pt (repli train all). Copies des deux sur le Mac de Clément et dans ma VM. Supprimables pour le quota : clem_A03r2_g4.pt, clem_A03r3_g4.pt, clem_A005_g2.pt, clem_A03r4_g4_last6.pt, et les checkpoints de mes essais c2 / e2 une fois lus.

## Ma file
Retiré : e2_clement_A03r6_g4_last6 (famille plafonnée). En cours sur gpu0 : c2 (rpow 5 train all, lecture seule), puis c3. Supprimez-les si Model Equality réclame le GPU 0.
