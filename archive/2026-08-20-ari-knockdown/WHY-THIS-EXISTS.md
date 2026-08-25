# Archived: the single-equation Adhesion Retention Index

`ARI(t) = clip(1 - k * w_bondline(t), 0, 1)`

Removed 2026-08-20 when the project was re-scoped as a student research study of
moisture *transport*. The scope note for that work is explicit: the dashboard
must not report a predicted adhesion strength without experimental data or a
validated adhesion model, and this index --- however simple and however carefully
caveated --- is a fabricated performance number.

It was replaced by an explicit **moisture threshold** criterion: the reported
quantity is the time for moisture at the bondline to reach a user-chosen level,
described as a proxy for potential performance change rather than as adhesion.

Nothing of substance is lost. The knockdown was strictly monotone in bondline
moisture, so "time to ARI threshold" and "time to moisture threshold" were
already the same instant computed two ways; the change removes the unjustified
mapping from moisture to a strength-like percentage, and with it the temptation
to read the number as a strength.

Restore only alongside experimental calibration of `k`.
