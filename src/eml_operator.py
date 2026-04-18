"""EML operator + arbitrary-precision Pi utilities.

The EML tree is the *symbolic, non-differentiable* back end. mpmath supplies
arbitrary-precision arithmetic. The single entry point we use from training /
inference is `eml_pi(dps)` and the reference `true_pi(n_digits)`.
"""
from __future__ import annotations

import mpmath


class EMLNode:
    def evaluate(self):
        raise NotImplementedError


class Constant(EMLNode):
    def __init__(self, value):
        if isinstance(value, complex):
            self.value = mpmath.mpc(value)
        else:
            self.value = mpmath.mpf(value)

    def evaluate(self):
        return self.value


class EML(EMLNode):
    """EML(x, y) = exp(x) - ln(y)."""

    def __init__(self, left: EMLNode, right: EMLNode):
        self.left = left
        self.right = right

    def evaluate(self):
        x = self.left.evaluate()
        y = self.right.evaluate()
        return mpmath.exp(x) - mpmath.log(y)


def _digits_only(mpf_value, n_sig: int) -> str:
    """Return the first n_sig significant digits of a positive real mpf as
    a plain string '3141592...'. No leading '3.'."""
    s = mpmath.nstr(mpf_value, n_sig, strip_zeros=False)
    if "e" in s or "E" in s:
        s = mpmath.nstr(mpf_value, n_sig, strip_zeros=False)
    s = s.replace(".", "").replace("-", "")
    return s[:n_sig]


def eml_pi(dps: int) -> str:
    """Compute Pi as a digit string using the EML identity ln(-1) = i*pi.

    Precision `dps` controls the mpmath working precision. Returns the first
    `dps` significant digits as a digit string (no decimal point).
    """
    dps = max(int(dps), 1)
    mpmath.mp.dps = dps + 5  # small guard so the last digit is trustworthy
    pi_val = mpmath.im(mpmath.log(-1))
    return _digits_only(pi_val, dps)


def true_pi(n_digits: int) -> str:
    """mpmath reference Pi, truncated to `n_digits` significant digits."""
    n_digits = max(int(n_digits), 1)
    mpmath.mp.dps = n_digits + 5
    return _digits_only(mpmath.pi, n_digits)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


if __name__ == "__main__":
    for n in (5, 50, 1000):
        ref = true_pi(n)
        got = eml_pi(n)
        print(f"N={n:5d}  match={common_prefix_len(ref, got)}  head={got[:20]}")
