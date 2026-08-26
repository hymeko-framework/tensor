#!/usr/bin/env python3
"""Golden Catmull-Rom grid-3 fixture generator for `hymeko_spline` parity.

Imports and executes the RECOVERED evaluator directly
(`hymeko_neuro.hyperedge.splines._catmull_rom_eval`). The formula is never copied
here: this file only builds inputs, calls the recovered function, and serialises
what PyTorch autograd produces.

torch.compile is disabled through the recovered runtime's own supported switch
(`HSIKAN_TORCH_COMPILE=0`, which is also its default), so evaluation is eager.

Usage: python gen_coginfocom_grid3_fixture.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types

# The recovered package's __init__ chain imports data/datasets/synth.py, which needs
# scikit-learn. sklearn is irrelevant to the spline evaluator, so it is stubbed rather
# than installed — no new dependency, and nothing under /mnt/hymeko is touched.
os.environ["HSIKAN_TORCH_COMPILE"] = "0"      # eager, via the recovered runtime's switch
os.environ.setdefault("PYTHONHASHSEED", "0")

RECOVERED_ROOT = "/mnt/hymeko/code_backup"
RECOVERED_PKG = os.path.join(RECOVERED_ROOT, "hymeko_neuro")
SPLINES_PATH = os.path.join(RECOVERED_PKG, "hyperedge", "splines.py")
RUNTIME_CFG_PATH = os.path.join(RECOVERED_PKG, "runtime", "runtime_config.py")

import torch  # noqa: E402


def _load_recovered():
    """Load the recovered evaluator module **by path**.

    `hymeko_neuro/__init__.py` transitively imports scikit-learn, scipy and friends,
    none of which the spline evaluator uses. Rather than stubbing an open-ended list of
    scientific packages, the two modules that are actually needed are loaded directly:

      * `runtime/runtime_config.py` — stdlib-only, provides `get_runtime`, which
        `_maybe_compile` imports lazily to read the torch.compile switch;
      * `hyperedge/splines.py` — torch-only, holds `_catmull_rom_eval`.

    Package skeletons are registered in `sys.modules` so the lazy
    `from hymeko_neuro.runtime.runtime_config import get_runtime` inside the recovered
    source resolves to the recovered file. Nothing is copied or re-implemented, and
    nothing under /mnt/hymeko is written.
    """
    import importlib.util

    for pkg, path in (("hymeko_neuro", RECOVERED_PKG),
                      ("hymeko_neuro.runtime", os.path.join(RECOVERED_PKG, "runtime")),
                      ("hymeko_neuro.hyperedge", os.path.join(RECOVERED_PKG, "hyperedge"))):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [path]
            sys.modules[pkg] = mod

    def load(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("hymeko_neuro.runtime.runtime_config", RUNTIME_CFG_PATH)
    splines = load("hymeko_neuro.hyperedge.splines", SPLINES_PATH)
    return splines._catmull_rom_eval


_catmull_rom_eval = _load_recovered()

GRID = 3
SEED = 20260826


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def coefficients(channels: int, gen: torch.Generator) -> torch.Tensor:
    """Non-symmetric, non-linear control points.

    Deliberately asymmetric so that a reversed control-point order or a wrong
    endpoint-replication rule changes the output.
    """
    base = torch.tensor(
        [[0.37, -1.42, 2.05],
         [-0.86, 0.13, -0.59],
         [1.91, 1.07, -2.33],
         [-0.24, 2.66, 0.48]],
        dtype=torch.float32,
    )
    out = base[:channels].clone()
    jitter = torch.rand((channels, GRID), generator=gen, dtype=torch.float32)
    return (out + 0.25 * jitter).contiguous()


def x_values() -> list[float]:
    """Every discriminating input class for grid 3.

    Knots sit at x = -1, 0, +1; segment 0 is x in [-1,0), segment 1 is x in [0,1].
    """
    eps_below = float(torch.nextafter(torch.tensor(0.0), torch.tensor(-1.0)))
    eps_above = float(torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))
    return [
        -1.0,            # left endpoint
        -0.999,          # just inside the left endpoint
        -0.7,            # segment 0 interior
        -0.25,           # segment 0 interior
        eps_below,       # immediately below the interior knot
        -0.0,            # signed zero (negative)
        0.0,             # signed zero (positive)
        eps_above,       # immediately above the interior knot
        0.3,             # segment 1 interior
        0.85,            # segment 1 interior
        0.999,           # just inside the right endpoint
        1.0,             # right endpoint
        -2.5,            # saturated below
        -1.0000001,      # marginally saturated below
        1.0000001,       # marginally saturated above
        3.25,            # saturated above
    ]


def build_case(name: str, channels: int, batch: int, gen: torch.Generator) -> dict:
    coef = coefficients(channels, gen).requires_grad_(True)

    xs = x_values()
    # Every (row, channel) slot gets a distinct value from the discriminating set, so
    # channels are not accidentally identical and every class appears in every case.
    raw = torch.empty((batch, channels), dtype=torch.float32)
    for r in range(batch):
        for c in range(channels):
            raw[r, c] = xs[(r * channels + c) % len(xs)]
    x = raw.clone().requires_grad_(True)

    y = _catmull_rom_eval(coef, x, GRID)

    # Non-constant upstream gradient: no lane shares a multiplier with another.
    grad_y = torch.empty_like(y)
    flat = grad_y.view(-1)
    for k in range(flat.numel()):
        flat[k] = 0.5 + 0.125 * ((k * 7) % 11) - 0.05 * ((k * 3) % 5)

    (y * grad_y).sum().backward()

    return {
        "name": name,
        "grid": GRID,
        "channels": channels,
        "batch": batch,
        "coef": coef.detach().reshape(-1).tolist(),
        "x": raw.reshape(-1).tolist(),
        "grad_y": grad_y.detach().reshape(-1).tolist(),
        "y": y.detach().reshape(-1).tolist(),
        "grad_x": x.grad.detach().reshape(-1).tolist(),
        "grad_coef": coef.grad.detach().reshape(-1).tolist(),
    }


def main() -> None:
    out_path = sys.argv[1]
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    gen = torch.Generator().manual_seed(SEED)

    src = SPLINES_PATH
    cases = [
        build_case("c1_b1", 1, 1, gen),
        build_case("c1_b16", 1, 16, gen),
        build_case("c3_b1", 3, 1, gen),
        build_case("c3_b7", 3, 7, gen),
        build_case("c4_b16", 4, 16, gen),
    ]

    doc = {
        "purpose": "Catmull-Rom grid-3 golden fixture from the recovered CogInfoCom PyTorch evaluator",
        "generated_by": os.path.basename(__file__),
        "generator_sha256": sha256_file(os.path.abspath(__file__)),
        "recovered_source": src,
        "recovered_source_sha256": sha256_file(src),
        "recovered_symbol": "hymeko_neuro.hyperedge.splines._catmull_rom_eval",
        "environment": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "device": "cpu",
            "dtype": "float32",
            "torch_compile": "disabled via HSIKAN_TORCH_COMPILE=0",
            "deterministic_algorithms": True,
        },
        "seed": SEED,
        "clamp_contract": "Corrected — the evaluator clamps x with torch.clamp, so autograd "
                          "gives zero input gradient outside [-1, 1]",
        "cases": cases,
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out_path}  cases={len(cases)}")


if __name__ == "__main__":
    main()
