# Constrained minimisation and forward-mode duals move into hymeko_tensor

Date: 2026-11-01. Authority: the user approved moving the numerical solver of transformation's
`hymeko_optimize` into the tensor repository ("AL + multistart + dual → tensor", answered "igen").

## Why

Transformation projects IR; it does not own numerical solvers. `hymeko_optimize::backend` carried
an augmented-Lagrangian solver with a bound-projected Newton inner solve and a private Gaussian
elimination, and `hymeko_optimize::dual` forward-mode dual numbers. Neither exists elsewhere in the
framework: Tensor has the canonical small dense solve (`linalg::dense::solve_in_place`) but no
constrained minimiser, Nagare's differentiable operators serve training graphs.

## What changes in hymeko_tensor

- `dual`: `Dual` moved unchanged (value plus gradient, constant/variable constructors, arithmetic).
- `optimize`: `Point`, the `Constrained` trait (dimension, bounds, point), `AugmentedLagrangian`
  with the same public fields and defaults, `Solution`, `SolveError`. The algorithm is unchanged
  except that its Newton systems call `solve_in_place` instead of the private elimination.
- No new dependency; the crate stays dependency-free in its default build.

## What stays in transformation

`read`, `problem`, `network` (the MNA reading that builds the constraints), `report`, the
`Backend` seam, `multistart`/`starts` (a geometric lattice chosen for the projection's variables),
`web` and `bin/solve`. `Problem` implements `Constrained`; transformation builds its evidence report
from the returned point. The SciPy subprocess backend (`External`, `backends/slsqp.py`) is removed.

## Numerical contract

Answers agree with the former in-place solver to the solver tolerance, not bit for bit: the
singular-pivot threshold (relative `64 ε max|a|` instead of absolute `1e-300`) and the
back-substitution summation order differ. No published result uses this solver.

## Checks

`cargo test -p hymeko_tensor --test optimize` (duals, an equality-constrained quadratic, an active
inequality, an active bound, evaluation failure, exhausted budget); in transformation the existing
`hymeko_optimize` oracle and authority tests through the new seam.
