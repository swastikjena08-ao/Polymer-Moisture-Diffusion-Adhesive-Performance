# Archived: the full multi-mechanism model

Snapshot taken 2026-08-20, immediately before the framework was deliberately
simplified. Kept because the simplification was requested "for now" and this
directory is the only copy — the project is not a git repository, so deleting
these files would otherwise be irreversible.

At the point of this snapshot: **267 tests passing**.

## What was removed from the live code

| Feature | Lived in |
|---|---|
| Irreversible interfacial hydrolysis (Arrhenius damage integral) | `adhesion.py` |
| Reversible plasticization knockdown | `adhesion.py` |
| Wet glass-transition depression (linear and Fox models) | `materials.py`, `adhesion.py` |
| Thermal softening factor near the wet Tg | `adhesion.py` |
| Three-mechanism ARI product and mechanism attribution | `adhesion.py` |
| Thermodynamic work-of-adhesion screening | `adhesion.py` |
| Concentration-dependent diffusivity `D(C)` and its Picard iteration | `diffusion/fd.py` |
| Freundlich humidity exponent and `rh_ref` on the isotherm | `sorption.py` |

## What was kept and still lives in `src/`

Fickian transport (exact series and finite volume), the one-sided/two-sided
geometry mapping, Arrhenius `D(T)`, humidity schedules, surface mass-transfer
resistance, sweeps, sensitivity analysis, the dashboard, and the CLI.

## Restoring a piece

These files are a working tree, not a diff. To bring one feature back, lift the
relevant block out of the file here and re-wire it — the module boundaries are
unchanged, so `adhesion.py` here drops into place against the current
`simulate.py` seam with only the `evaluate()` call signature to reconcile.

Read `docs/superpowers/specs/2026-08-19-hygro-adhesion-framework-design.md` in
the live tree for the physics and the references behind each removed mechanism.
