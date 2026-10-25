"""extract_error_signals: pull versions, error lines, stack traces and k8s status reasons from text."""
import re

from triage.contracts import Signals

K8S_VERSION_RE = re.compile(r"\bv1\.\d{1,2}\.\d{1,2}(?:[-+][\w.]+)?\b")
ERROR_LINE_RE = re.compile(
    r"\b(error|errors|failed|failure|panic|fatal|unable to|cannot|can't|forbidden|"
    r"unauthorized|timeout|timed out|denied|refused|not found)\b",
    re.IGNORECASE,
)
K8S_REASON_RE = re.compile(
    r"\b(CrashLoopBackOff|ImagePullBackOff|ErrImagePull|OOMKilled|Evicted|Pending|"
    r"ContainerCreating|FailedScheduling|FailedMount|NodeNotReady|Terminating|"
    r"CreateContainerConfigError|Unschedulable)\b"
)
TRACE_START_RE = re.compile(r"^(panic:|goroutine \d+ \[|Traceback \(most recent call last\):)")

MAX_ERROR_LINES = 15
MAX_TRACES = 3
TRACE_LINES = 12
MAX_LINE_CHARS = 300


def _dedupe(items):
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_error_signals(text: str) -> Signals:
    """Return structured signals found in free text. Pure function, no network."""
    lines = text.splitlines()

    versions = _dedupe(K8S_VERSION_RE.findall(text))
    reasons = _dedupe(K8S_REASON_RE.findall(text))

    error_lines = _dedupe(
        line.strip()[:MAX_LINE_CHARS]
        for line in lines
        if line.strip() and ERROR_LINE_RE.search(line)
    )[:MAX_ERROR_LINES]

    traces = []
    for i, line in enumerate(lines):
        if TRACE_START_RE.match(line.strip()):
            block = [l.rstrip()[:MAX_LINE_CHARS] for l in lines[i : i + TRACE_LINES]]
            traces.append("\n".join(block))
            if len(traces) == MAX_TRACES:
                break

    return {
        "versions": versions,
        "k8s_reasons": reasons,
        "error_lines": error_lines,
        "stack_traces": traces,
    }
