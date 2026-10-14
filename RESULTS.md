# RESULTS

Hypothesis under test: solving the gradient-matching QP on random **chimera** surrogate
models (bank checkpoints of known architectures + Rademacher fill) yields a solution `w`
(over the 90 class-pairs) that stays good on a victim of **unknown** architecture, because
the problem depends on architecture only through statistics in the common 90-pair space.

Toy problem: CIFAR-10, source 9 → target 4, fixed sinusoidal-stripe trigger (L∞ ≤ 8/255,
not optimized), λ_p = 0.05, polytope budget β ∈ {0.03, 0.01, 0.1}. Metric:
`eff(w) = (1 − B2_V(w)) / (1 − B2_V(w*_V))` (1 = oracle, 0 = no gain, <0 = worse than nothing).

## Decision criteria (fixed in advance)
- **Supported**: eff(chimeras) significantly > eff(naive transfer) and ≥ eff(ensemble of real
  checkpoints), and clearly above the 9→4 heuristic and the nulls (q=0 / random init).
- **Weak**: eff(chimeras) ≈ eff(random init) → inherited content adds nothing.
- **Useless**: eff(9→4 heuristic) ≈ eff(oracle) → problem too easy to discriminate; need a
  harder target.
- **Failed**: V3 (out-of-family) far below V1 and V2 → coverage depends on the grammar.

---

## E0 — generator validity (go/no-go)
**GO.**
- (a) 200 chimeras: forward+backward OK; `count_params == P(G)` for all; 200/200 within the
  ±10% band of the target N = 343,186.
- (b) identity reconstruction (q=1, identity injections, own checkpoint as donor): max|Δ| over
  blocks+head = 0.00e+00 — bit-exact.
- (c) ρ (inherited) mean ≈ 0.367, ρ_coh (coherent) mean ≈ 0.269; realized N mean ≈ 341,157
  (target 343,186). Histograms: `figures/E0_histograms.png`.

*(E0 uses the generator mechanics only — no training data needed.)*


## E1 — are the statistics comparable across architectures?
Epoch 15. Similarity group means:

| regime | cos(Q~) | rankcorr(c~) |
|---|---|---|
| real-real same arch | 0.967 ± 0.042 | 0.963 ± 0.029 |
| real-real other arch | 0.829 ± 0.122 | 0.902 ± 0.056 |
| chimera-victim | 0.782 ± 0.122 | 0.726 ± 0.159 |

Chimera vs real-other-arch closeness to each victim (cos Q~):
- V1: chimera=0.812, real-other=0.748 → chimera closer
- V2: chimera=0.731, real-other=0.728 → chimera closer
- V3: chimera=0.803, real-other=0.862 → chimera NOT closer

Figure: `figures/E1_similarity.png`



## E2 — does the solution transfer? (central test)
Epoch 15, β=0.03, K=64. eff (mean ± std), cos(w,w*), top-10 overlap:

| victim | chimera | real-ens | naive | chim q=0 | rand-init | heuristic | uniform |
|---|---|---|---|---|---|---|---|
| V1 | +0.96 | +0.98 | +0.98 | +0.91 | +0.91 | +0.99 | +0.01 |
| V2 | +0.81 | +1.00 | +0.99 | +0.44 | +0.43 | +0.98 | +0.03 |
| V3 | +0.94 | +0.95 | +0.99 | +0.88 | +0.88 | +1.00 | +0.01 |

Figure: `figures/E2_transfer.png`



## Decision
- **V1**: USELESS (heuristic ≈ oracle: target too easy)  (chimera=+0.96, real-ens=+0.98, naive=+0.98, heuristic=+0.99, q0=+0.91, rand-init=+0.91)
- **V2**: USELESS (heuristic ≈ oracle: target too easy)  (chimera=+0.81, real-ens=+1.00, naive=+0.99, heuristic=+0.98, q0=+0.44, rand-init=+0.43)
- **V3**: USELESS (heuristic ≈ oracle: target too easy)  (chimera=+0.94, real-ens=+0.95, naive=+0.99, heuristic=+1.00, q0=+0.88, rand-init=+0.88)



## E3 — ablations
eff at β=0.03, K=16 unless swept. Figure: `figures/E3_ablations.png`

- **q** (x=[0.0, 0.25, 0.5, 0.75, 1.0]):
    - V1: [0.91, 0.92, 0.98, 0.96, 0.97]
    - V2: [0.48, 0.54, 0.9, 0.86, 0.91]
    - V3: [0.89, 0.9, 0.95, 0.93, 0.94]
- **lambda** (x=[0.0, 0.3, 0.6, 1.0]):
    - V1: [0.96, 0.98, 0.97, 0.97]
    - V2: [0.77, 0.9, 0.88, 0.85]
    - V3: [0.94, 0.95, 0.95, 0.95]
- **K** (x=[1, 4, 16, 64]):
    - V1: [0.55, 0.96, 0.98, 0.96]
    - V2: [0.26, 0.84, 0.9, 0.75]
    - V3: [0.55, 0.93, 0.95, 0.93]
- **N** (x=[0.5, 1.0, 2.0]):
    - V1: [0.96, 0.98, 0.96]
    - V2: [0.82, 0.9, 0.84]
    - V3: [0.93, 0.95, 0.93]
- **t_single_vs_mixed** (x=[0, 1]):
    - V1: [0.98, 0.93]
    - V2: [0.9, 0.65]
    - V3: [0.95, 0.91]
- **coherent_tail** (x=[0, 1]):
    - V1: [0.92, 0.98]
    - V2: [0.55, 0.9]
    - V3: [0.91, 0.95]
- **bn_mode** (x=[0, 1]):
    - V1: [0.98, 0.54]
    - V2: [0.9, 0.95]
    - V3: [0.95, 0.65]

---

## Lecture des ablations E3
- **K** (ensemble) : effet majeur — K=1 médiocre (V2 0.26), saut dès K=4, optimum ≈ K=16, léger recul à 64.
- **q** (part héritée) : V2 passe de 0.48 (q=0) à ~0.90 (q≥0.5) → le contenu hérité compte.
- **tête cohérente** et **t unique** (vs mixte) : nettement meilleurs (V2 : 0.90 vs 0.55 / 0.90 vs 0.65).
- **λ** : optimum doux autour de 0.3 (défaut). **N cible** : optimum au N moyen de la banque.
- **BN** : `train` bien meilleur pour V1/V3, ~égal pour V2 → garder BN train.
Les ablations confirment que la structure héritée aide (cohérent avec E2 : chimère ≫ nulls),
mais cela reste dans un régime où la cible est trop facile (cf. verdict).

## Limites & recommandations
- **Cible trop facile** : `v = λ_p(∇𝓛_bd − g_c)` se projette presque entièrement sur la colonne
  (9→4) de G, donc l'heuristique triviale ≈ oracle. Pour un vrai test discriminant :
  optimiser le trigger, réduire λ_p, choisir une cible non alignée sur une seule paire
  (objectif multi-paires, ou contrainte `w ⟂ e_{9→4}`).
- **Simplification du générateur** : les donneurs sont pris à l'époque 15 (t fixe) au lieu d'un
  tirage t∼exponentielle par chimère ; l'ablation « t unique vs mixte » couvre l'esprit du facteur.
  Sans effet sur le verdict (qui tient à la facilité de la cible).
- **Famille B faible** (ConvNeXt SGD ≈ 64 %) : donneurs/V2 bruités ; explique la plus grande
  variance de V2, pas le verdict.
- **Couverture de grammaire** : V3 (hors-vocabulaire) déjà en retrait en E1 — conforme au critère
  « couverture dépend de la grammaire ».

---

# Étude 2 — empoisonnement clean-label par gradient matching (surrogate chimère)

Nouveau problème : backdoor source 9 → cible 4, trigger fixe. On optimise des perturbations Δ
(clean-label, ‖Δ‖∞≤ε) sur un sous-ensemble d'images de la **classe cible** pour que le gradient
d'entraînement empoisonné s'aligne avec le gradient backdoor, de sorte qu'une victime entraînée
*from scratch* acquière la backdoor. Hypothèse visée : un **unique** Δ crafté sur un ensemble de
**chimères** empoisonne plusieurs architectures in-family (universalité, sans connaître l'archi).

## P0 — pilote white-box (gate)  →  **NO-GO**
Craft white-box sur un checkpoint propre **convergé (époque 15)** de l'archi victime elle-même,
ε=16/255, 1 % du train empoisonné (500 img), victime from-scratch sans augmentation.

| victime | craft cos(g_tr,g_bd) | ASR poison | ASR null-trigger | ASR null-random | clean acc |
|---|---|---|---|---|---|
| V1 | 0.995 | 0.007 | 0.000 | 0.003 | 0.866 |
| V2 | 1.000 | 0.015 | 0.009 | 0.011 | 0.660 |

**Le matching réussit (cos≈1) mais la backdoor ne s'installe pas** : ASR(poison) ≈ ASR(nulls) ≈ 0.
Donc rien à transférer — inutile de passer aux chimères en l'état.

**Diagnostic.** Le craft se fait au θ **convergé** (perte propre ≈ 0). La victime, elle, part de
zéro : son update −η·g_tr(θ_t) ne pousse vers la backdoor que si g_tr≈g_bd **le long de la
trajectoire**, pas seulement au point final. On n'a apparié qu'à l'endpoint → l'effet ne se
réalise jamais tôt dans l'entraînement. (Le signe est correct ; les nulls confirment que le
trigger seul ne fuit pas → ce n'est pas le piège « trop facile ».)

**Correctif proposé (avant tout chimère)** : crafter contre un ensemble de checkpoints
**début/milieu de trajectoire** (époques 2/5/10, plusieurs graines) — déjà sur disque — voire
ré-entraîner le surrogate pendant le craft (recette Sleeper Agent). À valider en P0b.

## Pivot — on repart du code de référence (Witches' Brew)
Plutôt que de ré-implémenter le brewing (restarts, ensemble, augmentation différentiable) et
risquer les bugs du jouet P0, on branche la chimère sur le repo officiel
`JonasGeiping/poisoning-gradient-matching` (cloné dans `witches/`).
- **Type d'attaque** : *poisoning ciblé* natif du repo (faire mal-classer une image cible en
  classe adverse, clean-label) — c'est l'attaque qui marche en white-box. Pas une backdoor à
  trigger (= repo Sleeper Agent, éventuel 2e temps).
- **Chimère (phase 1)** : architecture tirée dans la grammaire (`sample_arch`+`assemble`),
  **entraînée from scratch** par le framework comme n'importe quel réseau ; l'héritage de poids
  de l'étude 1 viendra en phase 2.
- **Greffe** : `forest/victims/chimera_adapter.py` (importe `models.py`/`chimera.py` du projet) +
  branches `Chimera` / `V1,V2,V3,A,B` dans `get_model`. Transfert natif via `--vnet`
  (brasser sur `--net Chimera`, valider sur `--vnet V1,V2,V3`).
- **État** : le repo tourne ici (CIFAR réutilisé, mean/std identiques) ; dryrun
  brew-sur-chimères / valide-sur-V1 OK. Reste : gate d'efficacité white-box, puis matrice
  phase 1 (white-box / transfert naïf / ensemble réel / chimères).
