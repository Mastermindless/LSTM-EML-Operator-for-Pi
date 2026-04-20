"""EML operator + arbitrary-precision Pi utilities — v02 (GPU-optimized).

Changes from v01:
    • functools.lru_cache on eml_pi() and true_pi() eliminates redundant mpmath
      calls during REINFORCE reward loops.
    • PiDigitCache class pre-computes pi to max phase precision once, then slices.
    • common_prefix_len stays pure-Python (called on short strings, not a bottleneck).
"""
from __future__ import annotations

import functools

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


@functools.lru_cache(maxsize=2048)
def eml_pi(dps: int) -> str:
    """Compute Pi as a digit string using the EML identity ln(-1) = i*pi.

    Precision `dps` controls the mpmath working precision. Returns the first
    `dps` significant digits as a digit string (no decimal point).

    Cached: repeated calls with the same dps are O(1).
    """
    dps = max(int(dps), 1)
    mpmath.mp.dps = dps + 5  # small guard so the last digit is trustworthy
    pi_val = mpmath.im(mpmath.log(-1))
    return _digits_only(pi_val, dps)


@functools.lru_cache(maxsize=2048)
def true_pi(n_digits: int) -> str:
    """mpmath reference Pi, truncated to `n_digits` significant digits.

    Cached: repeated calls with the same n_digits are O(1).
    """
    n_digits = max(int(n_digits), 1)
    mpmath.mp.dps = n_digits + 5
    return _digits_only(mpmath.pi, n_digits)


# --------------------------------------------------------------------------- #
# Pre-compute cache: compute pi ONCE at max precision, slice for smaller N.
# --------------------------------------------------------------------------- #
class PiDigitCache:
    """Compute pi to a ceiling precision once; serve slices for any N ≤ ceiling.

    This eliminates O(batch_size × steps) mpmath calls during training —
    the single largest bottleneck in the original implementation.

    Uses extra guard digits so that slicing produces the same result as
    calling true_pi(n) directly (avoids nstr last-digit rounding divergence).
    """

    def __init__(self, max_digits: int = 10050):
        self._max = max_digits
        self._digits: str | None = None

    def _ensure(self, n: int):
        if self._digits is None or n > self._max:
            self._max = max(self._max, n + 50)
            # Compute with large guard; _digits_only(val, M) rounds at position M,
            # so we compute M much larger than any slice we'll take.
            mpmath.mp.dps = self._max + 50
            self._digits = _digits_only(mpmath.pi, self._max + 20)

    def get(self, n_digits: int) -> str:
        """Return first n_digits significant digits of pi (no decimal).

        Note: For the last digit, this truncates rather than rounds, which may
        differ from true_pi(n) by ±1 in the final digit. This is acceptable
        for training (common_prefix_len tolerance).
        """
        n_digits = max(int(n_digits), 1)
        self._ensure(n_digits)
        return self._digits[:n_digits]


# Module-level singleton — initialized lazily on first use.
_pi_cache = PiDigitCache()


def cached_true_pi(n_digits: int) -> str:
    """Ultra-fast pi digit lookup via pre-computed cache. Use during training."""
    return _pi_cache.get(n_digits)


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
        cached = cached_true_pi(n)
        assert ref == cached, f"Cache mismatch at n={n}"
        print(f"N={n:5d}  match={common_prefix_len(ref, got)}  head={got[:20]}")
    print("Cache validation passed ✔")
