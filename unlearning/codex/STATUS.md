<!-- CODEX_OOF_BEGIN -->
# Resume de nuit pour 7h40 — priorite OOF

Mise a jour : 22/11 07:35:07 CET. **Fin de surveillance07h35 ; consulter les resultats ci-dessus**.
Repli CPU2 vue propre (borne20min) +15profs : `GPU feature cache completed first; stop CPU fallback`.

- Plan UL_STRATEGIE : DINOv2-B14 gele, 5 plis x 3 partitions, pseudo-labels original, label smoothing0.1 ; cibles par vue et calibration moyenne/std CE.
- Statistiques unseen estimees par les probes retain-only6/7 ; evaluation par probe8 distinct, jamais utilise pour calibrer. Ce sont des approximations, pas les gold inconnus.
- FT blocs9-11 + head/norm, 1100pas, lr3e-5, KL retain poids2 + MSE logit centre ; 7 vues train. Shift/rotation/bruit reserves au test. Trois forces fixees d'avance : 1, 0.7, 1.4.
- Comparaison au meilleur checkpoint officiel disponible au lancement de chaque essai : `orehn/ckpt/pc_oof_k3d4_g4_last6.pt` actuellement. Refaire la comparaison sil change avant un envoi.
- Garde locale : retain_est>=0.75, amelioration proxy independant sur sous-ensembles30% (IC95 positif), KL des vues reservees amelioree. retain_est est accord/(1+KL) sur2000substituts sans labels, pas une accuracy mesuree.
- Soumissions Codex : aucune nouvelle depuis ft2_A03=0.0735. Ce coordinateur ne soumet jamais ; tout envoi exige controles actuels SUBMIT_LOCK, cooldown equipe10min et max3.
- CPU2 ; GPU2/3 uniquement si libres et files vides, via worker existant. Aucun sbatch/scancel, tmux, runner ME ou fichier de coequipier touche. Priorite2 A/image preparee et testee apres les trois forces OOF (4 essais locaux, plafond3 soumissions).

- `oof_calibrated_s1` : retain_est=0.7357, delta proxy IC95=[0.10202349722385406, 0.15482674539089203], candidat local admissible=False.
- `oof_calibrated_s07` : retain_est=0.8256, delta proxy IC95=[0.007204791065305471, 0.05122176557779312], candidat local admissible=True.
- `oof_calibrated_s14` : retain_est=0.6816, delta proxy IC95=[0.1782011240720749, 0.235918328166008], candidat local admissible=False.

Arret borne a07h35. Le precedent watcher repair est arrete via NIGHT_STOP. Artefacts : oof_features.pt, oof_targets.pt, oof_calibration.json, *_oof_eval.json, *_gate.json ; seuls les checkpoints finis sont utilisables.
<!-- CODEX_OOF_END -->

# Nouveau plan OOF — 22/11 01:18 CET

Prio1 en cours : nouveaux scripts oof_extract.py, oof_fit.py, oof_train.py. DINOv2-B14 officiel telecharge; features a extraire via worker. Ancien professeur OOF trop confiant (CE0.440/std0.615). 3 partitions de 5 plis, cibles par vue, calibration de deux moments; probes6/7 pour estimer unseen, probe8 reserve evaluation (ce ne sont pas des gold). Tests alteres shift/rotate/noise exclus de training. Controle retain public 2000 images independantes; estimation accord/(1+KL), pas accuracy etiquetee.

Dernier score confirme par journal equipe : clem_A03r3_g4 **0.2019**; Codex A03 **0.0735**. Aucun nouvel envoi Codex. Le plan precedent repair est arrete par fichier NIGHT_STOP.

<!-- CODEX_NIGHT_BEGIN -->
# Résumé de nuit pour Orehn — 7h40

Dernière mise à jour : 21/11 23:49:57 CET. État : GPU3 occupé ; attente.

- Meilleur score officiel d'équipe connu : 0.1219 (orehn/ckpt/clem_A005_g2.pt, 21/11 23:10). Le public ne décide pas des micro-variantes ; delta <0,01 ignoré.
- Soumissions Codex pendant le mode nuit : **aucune**. Avant la nuit : A03 soumis avec accord explicite, score 0,0735.
- Sécurité : contrôle public sans vrais labels ; accord et `accord/(1+KL)` ne prouvent pas une accuracy >=75%. Aucun candidat final certifié à ce stade. Les résultats officiels positifs prouvent le seuil serveur de70% sur ces envois, pas une marge locale de75%.
- Piste Codex : réparation du retain après oubli, ancrée aux prédictions du modèle source sur **7 vues** (propre, bright, flip, crop, décalage, bruit, blur). Deux configurations préfixées par source ; trois sources maximum, aucun balayage du score public.
- GPU3 uniquement, dépôts lorsque libre et file vide. GPU2 réservé aux collègues. **Jamais de sbatch/scancel**, aucun tmux/runner Model Equality touché ; aucun token lu par les expériences.
- Ce qui reste : obtenir une validation labellisée pertinente ou une estimation d'accuracy justifiée avant toute soumission autonome ; revoir les compromis locaux ci-dessous. Les meilleurs scores publics seuls ne suffisent pas.

- Expériences de réparation en préparation / attente GPU3.

Artefacts : `codex/logs/*_gate.json`, `*_done.json`, logits multivues, `COORD/RESULTS.csv`. Résumé de la méthode : `codex/NIGHT_PLAN.md`. Le coordinateur s'arrête au plus tard à 2026-11-22T07:35:00+01:00 ; il ne relance aucun nœud.
<!-- CODEX_NIGHT_END -->

# Codex Unlearning

21/11 vers 22h20 Paris — préparation en cours.

- CLAUDE.md, les quatre fichiers COORD, le rapport VPS et les scripts existants lus.
- Diagnostic détaillé : DIAGNOSTIC.md. Le score 0,01 n'identifie pas à lui seul le facteur limitant ; ne pas déduire les moments gold du seul meilleur candidat.
- Deux pistes : professeur appris uniquement sur contrôle public avec features précoces ; préférence négative bornée sans professeur forget.
- Scripts : common.py, retain_teacher.py, train.py. Tests GPU via worker existant seulement, en attendant une place libre sur GPU 2 ou 3.
- Aucune soumission, aucun sbatch/scancel, aucun token lu, aucune session tmux touchée.

Prochain point dans 30 minutes au plus tard pendant le travail actif.

21/11 22:14 Paris — lancement worker GPU 2, apres observation libre. File : 80_codex_initial.sh. Reference originale puis NPO et professeur retain-only. Aucun changement aux files ou travaux des autres.

21/11 22:17 — record officiel equipe maintenant 0,0572 (Orehn ft2_A045, RESULTS/SUBMISSIONS). Codex NPO initiale rejetee : CE9,89/std9,80 propre et22,26/std8,69 bright. Professeur retain-only prepare, cible implique CE2,50/std2,21 ; distillation en cours. Variante NPO stabilisee preparee, attend GPU libre. Reference originale reproduite : CE0,000322/std0,00243, bright0,61364/std1,70889 et KL controle~0.

21/11 22:20 — premier lot termine, 2 checkpoints verifies. NPO initiale rejetee (CE propre9,89, bright22,26). Professeur retain-only distille : CE propre0,828/std0,987 ; bright1,883/std1,580 ; KL controle0,0745, accord0,8705. Deux suites preparees : NPO avec restauration et professeur dynamique sur chaque vue, fine-tuning complet. GPU2 et3 occupes par collegues ; aucun nouveau depot tant que pas libre. Voir REPORT.md.

21/11 22:22 — suivi actif, GPU2/3 occupes. Un helper CPU attache attend leur liberation, sans scheduler ni controle de processus ; il deposera un seul script Codex si le STATUS est recent et la file vide (delai maximal15min). Professeur retain-only :99 classes predites, 37% de masse sur classes absentes de forget. A07 ancien reduisait la perte bright malgre oubli propre : asymetrie ajoutee au diagnostic.

21/11 22:28 — PRIORITE Orehn 22h23 : exception de soumission accordee pour ft2_A03 seulement. Controle cooldown 10min conservateur a22h27 ; lock pose, fichier float32/fini/strict=True 343629249 octets, SHA256 identique VPS/JURECA. Soumission unique lancee depuis VPS via run_submit.py existant, attend resultat. La cle est lue en memoire uniquement par ce script ; jamais affichee/copiee.

21/11 22:29 — Codex : GPU 2 observe libre (STATUS frais, aucune attente), depot 85_codex_followup.sh.

21/11 22:30 — SOUMISSION PRIORITAIRE TERMINEE : ft2_A03 obtient **0,0735**, precedent0,0572, improved=True. Journaux SUBMISSIONS.md et RESULTS.csv renseignes ; sortie officielle conservee dans codex/submission_ft2_A03_result.txt/json. Verrou de cette soumission retire apres journalisation. Aucune autre soumission autorisee ni lancee. Suite experimentale Codex demarree par worker sur GPU2 a22h29 (85_codex_followup), sorties codex/logs et ckpt ; evaluations et RESULTS automatiquement, aucun acces leaderboard dans ces scripts.

21/11 22:55 — ANTI_OVERFIT.md VPS lu et applique : pas de micro-tuning public (delta<.01 =bruit), selection locale multivue + KLcibles, aucun candidat final propose sans precision locale estimee>=75% et estimation documentee. Accord aigen non labellise !=accuracy ; A03 garde comme resultat officiel, pas certifie pour la marge. Voir ANTI_OVERFIT_APPLIED.md.
