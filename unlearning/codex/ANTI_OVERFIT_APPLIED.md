# Regles applicables depuis Orehn 21/11 22h55
Source : VPS ~/hackathon/results/ANTI_OVERFIT.md, lu 22h55.
- Classement final sur les 70% caches ; seule meilleure soumission publique retenue.
- Pas de micro-tuning public ; ecart strictement inferieur a0.01 traite comme bruit.
- Selection sur metriques locales : moyenne/std des pertes forget et copies alterees (flip/crop/decalage), KL vers cibles ; public confirme seulement une tendance.
- Aucun nouveau candidat final ni soumission proposee sans precision locale estimee >=75% et marge documentee.
- Ne pas confondre accord avec pseudo-label original sur forget, accord avec original sur controle sans labels, et accuracy de validation. Le controle aigen actuel n'a pas de vrais labels ; son accord ne certifie pas une accuracy de75%.
- A03 : score officiel0.0735, accord controle public0.817, KL0.1068 ; non certifie pour la marge75% en accuracy sans validation appropriee.
- Aucune autre soumission autorisee a Codex apres l'envoi unique d'A03.

Mise a jour mode nuit Orehn23h08 : autorisation conditionnelle de soumettre seul si precision locale>=75% justifiee, meilleure qualite locale que candidat actuel, SUBMIT_LOCK exclusif et10min depuis derniere soumission dequipe. Cette autorisation remplace la restriction anterieure a lenvoi unique A03 ; elle ne dispense daucun critere. En pratique aucun certificat accuracy disponible : scripts de nuit sans soumission. Interdiction utilisateur de sbatch/scancel conservee malgre exception plus generale dans NUIT.md.
