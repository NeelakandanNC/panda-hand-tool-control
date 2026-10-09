#!/usr/bin/env bash
# Fetch only the Franka Emika Panda arm and Shadow Hand models from MuJoCo Menagerie.
set -euo pipefail
cd "$(dirname "$0")/../assets"
[ -d menagerie ] && { echo "assets/menagerie already exists"; exit 0; }
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
git -C menagerie sparse-checkout set franka_emika_panda shadow_hand
