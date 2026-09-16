"""Uniform progress reporting for every script in this repo.

STANDING RULE: every long-running loop in this repo reports progress. A run on
Paperspace that prints nothing for forty minutes is indistinguishable from a
hung one, and the only way to find out is to kill it -- which is how two CelebA
sweeps were lost.

Two constraints shape this file.

1. NO NEW HARD DEPENDENCY. `tqdm` is used when it is importable and a built-in
   fallback is used when it is not. A missing package must never be the reason a
   twelve-hour GPU run does not start.

2. PROGRESS GOES TO STDERR, NEVER STDOUT. Several scripts here print markdown
   tables to stdout that get redirected into files
   (`python invariance.py > table.md`). A bar on stdout would corrupt them.

Also deliberate: updates are RATE-LIMITED by wall-clock, not by iteration count.
`gd_gpu.py` runs 2,000,000 iterations; formatting a bar on each one would cost
more than the arithmetic. Call `update()` as often as you like -- at most one
redraw every `min_interval` seconds actually happens.

Usage
-----
    from progress import pbar

    for x in pbar(items, desc="bundles"):           # wrap an iterable
        ...

    with pbar(total=T, desc="GD", unit="step") as b:  # manual, for tight loops
        for t in range(T):
            ...
            if t % 1000 == 0:
                b.update(1000)
            b.set_postfix_str(f"loss {loss:.4f}")

`set_postfix_str` and `update` are no-ops that cost a comparison when the bar is
disabled, so call sites need no `if verbose` guards.
"""

from __future__ import annotations

import os
import sys
import time

__all__ = ["pbar", "enabled"]


def enabled() -> bool:
    """Progress off when NO_PROGRESS is set. Honoured by every bar in the repo."""
    return not os.environ.get("NO_PROGRESS")


def _fmt_secs(s: float) -> str:
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class _Fallback:
    """Minimal tqdm-shaped bar, used when tqdm is not installed.

    Redraws in place on a TTY and prints one line per interval otherwise, so a
    piped log (`nohup python ... > log &`) stays readable instead of filling
    with carriage returns.
    """

    def __init__(self, iterable=None, total=None, desc="", unit="it",
                 min_interval=0.5, leave=True, disable=False):
        self.iterable = iterable
        self.total = total if total is not None else _guess_len(iterable)
        self.desc, self.unit = desc, unit
        self.min_interval, self.leave, self.disable = min_interval, leave, disable
        self.n = 0
        self.postfix = ""
        self._t0 = time.time()
        self._last = 0.0
        self._tty = sys.stderr.isatty()

    # -- tqdm-compatible surface ------------------------------------------
    def update(self, n: int = 1) -> None:
        self.n += n
        self._maybe_draw()

    def set_postfix_str(self, s: str) -> None:
        self.postfix = s
        self._maybe_draw()

    def set_description(self, s: str) -> None:
        self.desc = s
        self._maybe_draw()

    def close(self) -> None:
        if self.disable:
            return
        self._draw(final=True)
        if self._tty:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __iter__(self):
        if self.iterable is None:
            raise TypeError("pbar(total=...) is not iterable; use update()")
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    # -- drawing ----------------------------------------------------------
    def _maybe_draw(self) -> None:
        if self.disable:
            return
        now = time.time()
        if now - self._last >= self.min_interval:
            self._last = now
            self._draw()

    def _draw(self, final: bool = False) -> None:
        el = time.time() - self._t0
        rate = self.n / el if el > 0 else 0.0
        parts = [self.desc] if self.desc else []
        if self.total:
            pct = 100.0 * self.n / self.total
            filled = int(24 * self.n / self.total)
            parts.append(f"|{'#' * filled}{'.' * (24 - filled)}| "
                         f"{self.n}/{self.total} {pct:5.1f}%")
            if rate > 0 and not final:
                parts.append(f"eta {_fmt_secs((self.total - self.n) / rate)}")
        else:
            parts.append(f"{self.n} {self.unit}")
        parts.append(f"[{_fmt_secs(el)}]")
        if self.postfix:
            parts.append(self.postfix)
        line = "  ".join(parts)
        sys.stderr.write(("\r" + line + "\x1b[K") if self._tty else (line + "\n"))
        sys.stderr.flush()


def _guess_len(it):
    try:
        return len(it)
    except Exception:
        return None


def pbar(iterable=None, total=None, desc="", unit="it", min_interval=0.5,
         leave=True):
    """A progress bar: tqdm if available, the built-in fallback otherwise.

    Pass an iterable to wrap it, or `total=` and drive it with `update()`.
    Always writes to stderr. Disabled by the NO_PROGRESS environment variable.
    """
    disable = not enabled()
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _Fallback(iterable, total, desc, unit, min_interval, leave, disable)

    return tqdm(iterable, total=total if total is not None else _guess_len(iterable),
                desc=desc or None, unit=unit, leave=leave, disable=disable,
                mininterval=min_interval, file=sys.stderr, dynamic_ncols=True)
