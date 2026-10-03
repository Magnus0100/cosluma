# Appearance operator review

The first 100% crop check exposed two practical failure modes in the demo:

1. The original `DSC02634` polygon was a convex, low-point outline. It filled
   the empty wedge between the raised hand and the shoulder and also extended
   above the hair. When that alpha was used for compositing, bright reference
   pixels leaked into the set around the head and arm.
2. Fitting the affine operator only on the subject mask produced an almost
   identity mapping (`scale ≈ 1.05` on the DSC02634 run). The final retained
   the dark source exposure while the manual final used the brighter reference
   lighting. This is a fitting-policy issue, not a resolution issue.

The polygon backend now uses a tighter DSC02634 outline in
`scripts/make_mask.py`. A fresh run was generated with:

```powershell
& $py scripts/make_mask.py --sample DSC02634 --out-dir outputs/mask_tight --proxy-side 1024 --feather 32
& $py scripts/run_demo.py --sample DSC02634 --alpha outputs/mask_tight/DSC02634/alpha.png --out-dir outputs/operator_tight
```

The mask edge is materially better, but the operator-only-on-alpha result is
still too dark against `data/DSC02634.png`. For the next visual pass, keep the
external alpha for blending and fit lighting from the full aligned frame (or
from a separately chosen background trust region); do not let a subject-only
fit collapse the style transform toward identity.

Artifacts: `outputs/mask_tight/DSC02634/alpha.png` and
`outputs/operator_tight/DSC02634/comparison.jpg`.
