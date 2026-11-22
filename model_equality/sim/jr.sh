#!/usr/bin/env bash
# Exécute une commande sur JURECA.
#  - PC Windows : via le conteneur Docker « jureca-ssh » (le ssh de Git Bash ne sait pas multiplexer).
#  - VPS / Linux : ssh direct, connexion maître ouverte avec `ssh -fN jureca` (+ code MFA), voir ~/.ssh/config.
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx jureca-ssh; then
  MSYS_NO_PATHCONV=1 docker exec -i jureca-ssh ssh jureca "$@"
else
  ssh jureca "$@"
fi
