# Nuit du 21 au 22 novembre — Codex Unlearning

NUIT.md et ANTI_OVERFIT.md du VPS lus et appliqués. L'interdiction explicite de sbatch/scancel du dernier message utilisateur prime sur l'autorisation générale de relance contenue dans NUIT.md. Aucun contact bloquant avec Orehn endormi.

La piste est distincte des balayages A/gamma des collègues : **réparer le retain après l'oubli**, en préservant les distributions du checkpoint source sur chaque vue forget. Source = modèle officiel terminé, journalisé dans COORD. Deux configurations fixées à l'avance : quatre derniers blocs / neuf cents pas ; tous les blocs / mille deux cents pas. Trois sources au maximum pendant la nuit, une nouvelle source seulement après changement public >=0,01. Aucun choix d'hyperparamètre d'après un nouveau score public.

Contrôle local sur 3 500 forget et 2 000 contrôles publics réservés. Sept vues : clean, bright, flip, crop, shift avec réflexion, noise et blur. Conservation locale : KL(source||réparé) <=0,02 pour chaque vue, différence absolue moyenne/std CE <=0,15 et KL vers les cibles fixes non dégradée de plus de0,02. Retain amélioré : baisse de KL d'au moins0,01 et accord non décroissant. Ces tolérances sont fixées avant essais ; elles n'estiment pas la proximité au gold invisible.

Le filtre supplémentaire d'accord conservateur utilise la borne Wilson 95% de l'accord sur les contrôles, divisée par 1+KL. Il doit dépasser75% pour qu'une réparation soit signalée comme prometteuse. **Ce proxy n'est pas une accuracy validée** : les contrôles aigen n'ont pas de vrais labels. Les scripts ne soumettent donc rien automatiquement. Une estimation de précision valide >=75%, ainsi qu'une amélioration locale et le protocole de verrou/cooldown incluant Clément, restent requis avant tout envoi autonome. Pas de candidat final certifié tant que cette estimation manque.

Le coordinateur `night_watch.py` est un processus CPU léger : lecture de fichiers de coordination, dépôts GPU3 seulement quand le worker indique libre et file vide, résumé en tête de STATUS.md toutes les30 secondes. Il n'exécute pas de modèle sur le login, ne lance aucun job Slurm, ne touche aucune session tmux, ne lit aucun token. Il se termine à7h35 et n'essaie pas de réparer un nœud perdu. Tous les scripts GPU ont un délai maximal de30min.

Arrêt de surveillance réversible : créer `codex/NIGHT_STOP` (ne tue pas les expériences déjà confiées au worker). Aucun fichier de collègue supprimé. Les essais refusés conservent leurs métriques/logits ; les checkpoints ne sont sauvés que pour les réparations localement améliorées et conservatrices.

Verification 2026-11-21T23:16:02.046462: syntaxe compilee ; controle du filtre sur donnees synthetiques : amelioration retain et conservation forget admises, mais soumission refusee sans accuracy certifiee ; regression sur la vue shift rejetee. Coordinateur vivant confirme par ps. Essais GPU reels en attente des deux scripts de Clement (42/43), sans les interrompre.
