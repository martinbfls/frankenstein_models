# Frankenstein surrogates for gradient-based poisoning

Question : une attaque par empoisonnement à base de gradients (Witches' Brew, FLIP) forgée sur des
modèles **Frankenstein**, reconstruits à partir d'une banque de checkpoints d'architectures connues
et complétés par des poids aléatoires, transfère-t-elle vers une victime d'architecture **inconnue** ?
On compare trois bras : architecture connue (white-box), Frankenstein, mauvaise architecture.
Métriques : **ASR** et **similarité cosinus** entre le gradient des poisons et le gradient adverse.

## Lancer en local sur Mac (puce Apple / MPS)
Le device est choisi automatiquement (CUDA > MPS > CPU) et, sur MPS, le profil **`mac`** s'active :
mêmes bras et même protocole, mais réseaux plus petits (`frankenstein/config.py`), 8 époques, batch 256,
K=3, 2 clés Witches, 60 itérations de brew, 2 itérations FLIP. Checkpoints et résultats sont séparés
du profil cluster : `checkpoints_mac/` et `results/hypothesis_mac/` (le profil `full` est inchangé).
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python torch torchvision numpy scipy pandas matplotlib toml tqdm pillow
.venv/bin/python -m frankenstein.run_study --dry-run            # plan, n'exécute rien
.venv/bin/python -m frankenstein.run_study                      # tout, reprenable (relancer la même commande)
.venv/bin/python -m frankenstein.run_study --attacks witches --no-asr   # le plus léger : cosinus seul
```
Durées mesurées sur M4 Pro : banque + victimes ~20 min, un brew Witches + réentraînement d'une victime
~5 min, une optimisation de labels FLIP ~5 min. Étude complète : de l'ordre de 2 h ; `--no-asr`,
`--attacks witches|flip` ou `--keys 1` pour réduire. Variables : `FRANK_PROFILE=full|mac`,
`FRANK_DEVICE=cpu|mps|cuda`, `FRANK_DATA_DIR`. Si le téléchargement de CIFAR-10 (serveur de Toronto)
est très lent, placer `cifar-10-python.tar.gz` dans `data/`.

## Campagne complète sur GPU (CUDA), en une commande
```bash
python -m frankenstein.run_study --device cuda     # profil `full` automatique sur CUDA : banque, victimes, Witches, FLIP, figures
```
Après chaque unité terminée, la commande affiche `[i/N units] elapsed … | remaining ~…` ; le temps
restant est estimé à partir de la durée moyenne des unités déjà faites (conservée dans
`results/*/progress.json`, donc disponible dès la reprise). Tant qu'un type d'unité n'a jamais été
chronométré (premier lancement), l'estimation est une borne basse et le dit. `--device cpu|mps|cuda`
et `--profile full|mac` remplacent `FRANK_DEVICE` / `FRANK_PROFILE`.

## Contenu
- `frankenstein/` : config (banque A/B, victimes V1-V3), modèles, générateur (`chimera.py`),
  entraînement de la banque et des victimes (`train_bank.py`), auto-test du générateur (`check_generator.py`).
- `witches/` : Witches' Brew, avec les modèles `Frankenstein`, `Frankenstein-q<val>`, `V1..V3`, `A`, `B`
  (`forest/victims/frankenstein_adapter.py`).
- `flip/` : FLIP, avec les flags `frank-<seed>[-q<val>]`, `V1..V3`, `A`, `B`
  (`modules/base_utils/model/frankenstein.py`).

## Tout en une commande (depuis la racine du dépôt)
```bash
python -m frankenstein.run_study --dry-run   # affiche le plan et les commandes, n'exécute rien
python -m frankenstein.run_study             # tout (reprenable), puis figures + résumé
python -m frankenstein.run_study --plots-only
python -m frankenstein.run_study --attacks witches --no-asr   # le moins cher : cosinus seul
```
Bras : white-box (checkpoint de la victime), Frankenstein (K=4), Frankenstein q=0 (nul),
banque A, banque B (mauvaise architecture). Witches : `--keys` cibles, une infusion par
(bras, clé) évaluée sur V1-V3. FLIP : experts = points de départ fixes (Frankenstein ou
checkpoints entraînés), même protocole pour tous les bras. Sorties dans `results/hypothesis/` :
`figures/*.png`, `summary.md`, `logs/`.
**Reprise** : si la commande est interrompue (Ctrl-C, plantage, session perdue), relancer exactement
la même commande. Les unités terminées sont sautées (marqueurs dans `done/`) ; le poison infusé est
mis en cache (`witches/deltas/`), donc une interruption pendant le réentraînement des victimes ne
refait pas l'infusion, et seules les victimes manquantes sont évaluées. Seule l'unité en cours au
moment de l'interruption est refaite. Relancer avec d'autres paramètres est refusé (`run_args.json`),
sauf `--keys` plus grand (ajoute des cibles) ou `--force`.

## Étapes (depuis la racine du dépôt)
```bash
python -m frankenstein.train_bank all        # banque A,B + victimes V1-V3, 3 graines, 15 époques
python -m frankenstein.check_generator       # auto-test du générateur (CPU)
```

### Witches' Brew (depuis `witches/`)
```bash
# white-box
python brew_poison.py --net V1 --vnet V1
# Frankenstein : K=4 surrogates hérités de la banque, pas de ré-entraînement
python brew_poison.py --net Frankenstein --ensemble 4 --skip_clean_training --vnet V1
# mauvaise architecture (autre famille de la banque)
python brew_poison.py --net B --vnet V1
# nul : même grammaire, aucun poids hérité
python brew_poison.py --net Frankenstein-q0 --ensemble 4 --skip_clean_training --vnet V1
```
- ASR : `target_acc_reinit` dans `tables/` (victime ré-entraînée from scratch sur les données empoisonnées).
- Cosinus : `victim_cos` (moyenne sur les checkpoints propres `checkpoints/<vnet>_s*_e<align_epochs>.pt`),
  `victim_cos_clean` = même mesure sans perturbation (référence). Détail par checkpoint dans le log.
- Ne pas utiliser `--step` (chemin de ré-initialisation cassé dans le code d'origine) ni `dist_brew_poison.py`
  (pas de recalibration BN).

### FLIP (depuis `flip/`)
```bash
python run_experiment.py frankenstein_attack   # experiments/frankenstein_attack/config.toml
```
- Frankenstein : **pas de `train_expert`**. Dans `[generate_labels]`, sans `input_pths`/`opt_pths` et avec
  `expert_model = "frank-{}"`, l'expert k est le réseau `frank-k` reconstruit depuis la banque ; chaque pas
  expert/étudiant part de ces poids, avec un état d'optimiseur vierge (pas de momentum).
  `expert_config.experts = K`.
- `[train_user]` : `user_model = "V1"` (ASR = précision sur le jeu de test empoisonné).
- White-box : experts `V1` entraînés par `train_expert` (trajectoires, `input_pths`) ; mauvaise architecture :
  idem avec `B` ou `r32p`.
