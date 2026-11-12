"""
LogC4 correctness against the ARRI LogC4 specification, plus the ComfyUI node's
input-domain check.

Reference values come from the ARRI LogC4 specification (2022) as implemented by
colour-science (`colour.models.log_encoding_ARRILogC4` / `log_decoding_ARRILogC4`).
They are hard-coded so this test has no extra dependency.

Checks:
  1. tools/hdr_encodings.py and the ComfyUI node decode to the reference, including
     codes between 0 and C (= LogC4 black) and negative codes.
  2. The decode is continuous at C (no jump between branches).
  3. encode -> decode round-trips for scene-linear values >= t (the curve's toe).
  4. The node's domain check accepts LogC4-encoded images (incl. 8-bit training PNGs)
     and flags display-referred (sRGB) images fed to the LogC4 decoder.

Usage:  python3 test_logc4_reference.py      (exit code 0 = all passed)
"""
import importlib.util, os, sys, types
import numpy as np
from hdr_encodings import logc4_fwd, logc4_inv, _C4_c, _C4_t

HERE = os.path.dirname(os.path.abspath(__file__))

# (code E', scene-linear E) from the ARRI LogC4 reference implementation
DECODE_REF = [
    (-0.05, -0.023736856550440763),
    (0.0,   -0.01805699611991131),
    (0.02,  -0.01552371034468367),
    (0.05,  -0.010546845108110301),
    (0.09,  -0.0008652797833480117),
    (0.15,   0.024164133954121862),
    (0.6,    6.481583651144599),
    (1.0,  469.8),
]
ENCODE_REF = [
    (-0.02, -0.017104327684224208),
    (0.0,    0.09286412512218964),
    (0.18,   0.2783958365482653),
    (1.0,    0.42751936483530617),
    (100.0,  0.8553946933762028),
]


def load_node_module():
    """Import the node pack (repository root __init__.py) without ComfyUI: stub folder_paths (and torch if absent)."""
    sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")
    path = os.path.join(HERE, "..", "__init__.py")
    spec = importlib.util.spec_from_file_location("comfyui_krea_hdr_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def srgb_oetf(L):
    L = np.clip(L, 0.0, 1.0)
    return np.where(L <= 0.0031308, 12.92 * L, 1.055 * np.power(L, 1 / 2.4) - 0.055)


def main():
    node = load_node_module()
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    print("-- decode vs ARRI LogC4 reference --")
    V = np.array([v for v, _ in DECODE_REF]); ref = np.array([e for _, e in DECODE_REF])
    for impl_name, impl in (("hdr_encodings.logc4_inv", logc4_inv),
                            ("node.logc4_to_linear", node.logc4_to_linear)):
        err = np.abs(impl(V) - ref) / np.maximum(np.abs(ref), 1e-3)
        check(impl_name, err.max() < 1e-9, f"max rel err {err.max():.2e}")

    print("-- encode vs ARRI LogC4 reference --")
    L = np.array([l for l, _ in ENCODE_REF]); ref = np.array([v for _, v in ENCODE_REF])
    err = np.abs(logc4_fwd(L) - ref).max()
    check("hdr_encodings.logc4_fwd", err < 1e-12, f"max abs err {err:.2e}")

    print("-- continuity at LogC4 black (code C) --")
    eps = 1e-9
    for impl_name, impl in (("hdr_encodings.logc4_inv", logc4_inv),
                            ("node.logc4_to_linear", node.logc4_to_linear)):
        jump = abs(float(impl(np.array([_C4_c - eps]))[0]) - float(impl(np.array([_C4_c]))[0]))
        check(f"{impl_name} continuous at C", jump < 1e-8, f"jump {jump:.2e}")
        jump0 = abs(float(impl(np.array([-eps]))[0]) - float(impl(np.array([0.0]))[0]))
        check(f"{impl_name} continuous at 0", jump0 < 1e-8, f"jump {jump0:.2e}")

    print("-- round trip encode -> decode --")
    Ls = np.concatenate([np.linspace(_C4_t, 0.0, 101), np.geomspace(1e-6, 469.8, 1001)])
    rt = logc4_inv(logc4_fwd(Ls))
    err = (np.abs(rt - Ls) / np.maximum(np.abs(Ls), 1e-3)).max()
    check("round trip from toe t to ceiling", err < 1e-9, f"max rel err {err:.2e}")

    print("-- node input-domain check --")
    rng = np.random.default_rng(0)
    # synthetic scene-linear image: log-normal radiance around mid-gray with some highlights
    lin = np.exp(rng.normal(np.log(0.18), 1.6, size=(256, 256, 1))) * rng.uniform(0.6, 1.4, (256, 256, 3))
    logc4_code = np.clip(logc4_fwd(lin), 0, 1)
    st = node.logc4_domain_stats(logc4_code)
    check("LogC4 float codes accepted", st["looks_logc4"], f"below-black {100*st['frac_below_logc4_black']:.2f}%")
    st = node.logc4_domain_stats(np.round(logc4_code * 255) / 255)
    check("LogC4 8-bit (training PNG) codes accepted", st["looks_logc4"],
          f"below-black {100*st['frac_below_logc4_black']:.2f}%")
    st = node.logc4_domain_stats(srgb_oetf(lin / 4.0))     # same scene, display-referred
    check("display-referred sRGB codes flagged", not st["looks_logc4"],
          f"below-black {100*st['frac_below_logc4_black']:.2f}%")

    print(f"\n{'ALL PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
