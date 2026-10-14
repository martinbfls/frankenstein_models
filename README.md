# Frankenstein surrogates for gradient-based poisoning

Question : une attaque par empoisonnement à base de gradients (Witches' Brew, FLIP) forgée sur des
modèles **Frankenstein**, reconstruits à partir d'une banque de checkpoints d'architectures connues
et complétés par des poids aléatoires, transfère-t-elle vers une victime d'architecture **inconnue** ?
On compare trois bras : architecture connue (white-box), Frankenstein, mauvaise architecture.
Métriques : **ASR** et **similarité cosinus** entre le gradient des poisons et le gradient adverse.

## Contenu
- `frankenstein/` : config (banque A/B, victimes V1-V3), modèles, générateur (`chimera.py`),
  entraînement de la banque et des victimes (`train_bank.py`), auto-test du générateur (`check_generator.py`).
- `witches/` : Witches' Brew, avec les modèles `Frankenstein`, `Frankenstein-q<val>`, `V1..V3`, `A`, `B`
  (`forest/victims/frankenstein_adapter.py`).
- `flip/` : FLIP, avec les flags `frank-<seed>[-q<val>]`, `V1..V3`, `A`, `B`
  (`modules/base_utils/model/frankenstein.py`).

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
