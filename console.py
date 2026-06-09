"""In-process console capture for the web UI: tees stdout/stderr and the Python
logging system into a ring buffer that a collapsible UI sidebar can display, so
all telemetry, progress, warnings, and tracebacks are visible in the browser."""
import logging
import sys
import threading
import time
from collections import deque

_LOCK = threading.Lock()
_BUF = deque(maxlen=2000)
_installed = False


def _emit(line):
    line = (line or "").rstrip("\r\n")
    if not line.strip():
        return
    with _LOCK:
        _BUF.append(f"{time.strftime('%H:%M:%S')}  {line}")


class _BufHandler(logging.Handler):
    def emit(self, record):
        try:
            _emit(self.format(record))
        except Exception:
            pass


class _Tee:
    """Pass writes through to the real stream and also into the ring buffer.
    Carriage returns (tqdm bars) are normalized to newlines so progress shows."""
    def __init__(self, stream):
        self.stream = stream
        self._partial = ""

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        try:
            text = self._partial + s
            parts = text.replace("\r", "\n").split("\n")
            self._partial = parts.pop()
            for p in parts:
                _emit(p)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def log(msg):
    """Emit an explicit console line from app code."""
    _emit(str(msg))


def install():
    global _installed
    if _installed:
        return
    _installed = True
    h = _BufHandler()
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)
    _emit("=== console capture started ===")


def get_text(maxlines=600):
    with _LOCK:
        lines = list(_BUF)[-maxlines:]
    return "\n".join(lines)


def clear():
    with _LOCK:
        _BUF.clear()
    _emit("=== console cleared ===")
