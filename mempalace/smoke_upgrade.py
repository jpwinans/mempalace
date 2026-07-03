"""
smoke_upgrade.py — Live-MCP upgrade smoke-check (the post-restart gate).

Codifies this session's #1 lesson: ``diary_write`` was present in the code but
a *stale running server* failed to advertise it to the client, which surfaced
only as a silent capability gap. This check connects to a freshly-launched
server over the real stdio JSON-RPC wire (NOT a module import) and asserts the
upgrade actually took:

  1. ``serverInfo.version`` >= 3.4.0
  2. ``tools/list`` advertises every expected tool — load-bearing: ``mempalace_diary_write``
  3. ``diary_write`` -> ``diary_read`` round-trips (against a DISPOSABLE sandbox palace)
  4. ``search`` returns without raising

Guardrails:
  - version + advertise + search run **read-only** against the configured palace.
  - the diary round-trip writes ONLY to a throwaway sandbox palace
    (``MEMPALACE_PALACE_PATH=tmpdir``), so the configured palace gains no drawer
    and no ``col.delete`` is ever needed to clean up.

This is a POST-RESTART gate: it verifies the on-disk code is current and that
the configured launch advertises + runs the expected tools. It deliberately
does NOT try to introspect an already-bound client stdio subprocess — that is
client-owned and unreachable from a standalone process. The complementary
in-session "is diary_write in my live tool list?" check is an agent action that
lives in the upgrade runbook (A4).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Iterable

MIN_VERSION = "3.4.0"

# The full set of tools the server is expected to advertise. Pinned explicitly
# so the gate catches ANY expected tool silently disappearing — not only the
# one that bit us. ``mempalace_diary_write`` is the load-bearing member.
EXPECTED_TOOLS = frozenset(
    {
        "mempalace_add_drawer",
        "mempalace_apply_decay_pass",
        "mempalace_check_duplicate",
        "mempalace_compute_entity_tunnels",
        "mempalace_compute_hallways",
        "mempalace_create_tunnel",
        "mempalace_delete_drawer",
        "mempalace_delete_tunnel",
        "mempalace_diary_read",
        "mempalace_diary_write",
        "mempalace_find_tunnels",
        "mempalace_follow_tunnels",
        "mempalace_get_aaak_spec",
        "mempalace_get_drawer",
        "mempalace_get_taxonomy",
        "mempalace_graph_stats",
        "mempalace_hook_settings",
        "mempalace_kg_add",
        "mempalace_kg_invalidate",
        "mempalace_kg_query",
        "mempalace_kg_stats",
        "mempalace_kg_timeline",
        "mempalace_list_drawers",
        "mempalace_list_hallways",
        "mempalace_list_rooms",
        "mempalace_list_tunnels",
        "mempalace_list_wings",
        "mempalace_memories_filed_away",
        "mempalace_potentiate",
        "mempalace_reconnect",
        "mempalace_search",
        "mempalace_status",
        "mempalace_sync",
        "mempalace_traverse",
        "mempalace_update_drawer",
    }
)

# Throwaway agent for the round-trip. NOTE: mempalace's own sanitize_name
# (_SAFE_NAME_RE) rejects leading/trailing underscores, so the literal
# "__healthcheck__" is NOT a valid agent/wing name. The disposable sandbox
# palace is the isolation mechanism here; this name only needs to be valid and
# obviously a throwaway.
_HEALTHCHECK_AGENT = "smoke-healthcheck"


# ── pure logic ───────────────────────────────────────────────────────────


def _parse_version(v: str) -> list[int]:
    parts: list[int] = []
    for component in str(v).split("."):
        # Take only LEADING digits so a pre-release suffix is the base number,
        # not a misread patch: "3.4.0rc1" -> [3, 4, 0], not [3, 4, 1].
        m = re.match(r"\d+", component)
        parts.append(int(m.group()) if m else 0)
    return parts


def version_meets(reported: str, minimum: str = MIN_VERSION) -> bool:
    """True iff ``reported`` is >= ``minimum`` as a dotted numeric version.
    Shorter versions are zero-padded ("3.4" == "3.4.0")."""
    r = _parse_version(reported)
    m = _parse_version(minimum)
    width = max(len(r), len(m))
    r += [0] * (width - len(r))
    m += [0] * (width - len(m))
    return r >= m


def missing_required_tools(advertised: Iterable[str], required: Iterable[str]) -> list[str]:
    """Names in ``required`` that ``advertised`` does not contain, sorted."""
    adv = set(advertised)
    return sorted(t for t in required if t not in adv)


# ── result types ─────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class SmokeReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def format(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        verdict = "PASS" if self.ok else "FAIL"
        lines.append(f"SMOKE-CHECK: {verdict}")
        return "\n".join(lines)


# ── stdio JSON-RPC driver ────────────────────────────────────────────────


def _default_server_cmd() -> list[str]:
    """Launch the server exactly as a client would — through the real entry
    point over stdio. NOT an in-process import: this exercises the actual
    advertise path that a stale server gets wrong."""
    return [sys.executable, "-c", "from mempalace.mcp_server import main; main()"]


def _mcp_session(
    requests: list[dict],
    *,
    env: dict | None = None,
    server_cmd: list[str] | None = None,
    timeout: float = 90.0,
) -> dict:
    """Spawn the server, feed newline-delimited JSON-RPC requests over stdin,
    return ``{id: response}``. The server loops until EOF, so closing stdin
    after the last request drains every response."""
    server_cmd = server_cmd or _default_server_cmd()
    proc_env = dict(os.environ)
    # PYTHONPATH is popped by main() for children; the parent test env is fine.
    if env:
        proc_env.update(env)
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    try:
        proc = subprocess.run(
            server_cmd,
            input=payload,
            capture_output=True,
            text=True,
            env=proc_env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # A hung server must produce a clean FAIL report (no responses), not
        # crash the gate — the gate exists to detect a broken server.
        return {}
    responses: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # log noise / non-JSON-RPC lines are ignored
        if isinstance(obj, dict) and obj.get("id") is not None:
            responses[obj["id"]] = obj
    return responses


def _call(name: str, arguments: dict, req_id) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _result_text(response: dict) -> str:
    """Extract the text payload from a tools/call result, or '' on error."""
    try:
        return response["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


# ── palace read-only count (for the no-stray-drawer assertion) ───────────


def drawer_count(palace_path: str) -> int:
    """Read-only drawer count for a palace; 0 when the collection was never
    created. Never opens a write path, never triggers repair."""
    try:
        from mempalace.palace import get_collection
    except Exception:
        return 0
    try:
        col = get_collection(palace_path, "mempalace_drawers", create=False)
    except Exception:
        return 0
    if col is None:
        return 0
    try:
        return int(col.count())
    except Exception:
        return 0


# ── orchestration ────────────────────────────────────────────────────────


def _probe_token() -> str:
    # Confabulation-impossible: a fresh random token must round-trip verbatim,
    # so a PASS proves the write reached storage and the read returned it.
    return "smoke-probe-" + os.urandom(8).hex()


def run_smoke(
    palace_path: str | None = None,
    *,
    server_cmd: list[str] | None = None,
    sandbox_path: str | None = None,
    min_version: str = MIN_VERSION,
) -> SmokeReport:
    """Run the four-assertion smoke-check and return a structured report.

    ``palace_path`` — the configured palace to probe read-only (version /
    advertise / search). When ``None``, the server resolves its own configured
    palace from env/config.json (production gate usage).

    ``sandbox_path`` — disposable palace for the diary round-trip. When
    ``None``, a temp dir is created and removed automatically.
    """
    report = SmokeReport()

    # ── Session 1: read-only against the configured palace ──────────────
    real_env = {"MEMPALACE_PALACE_PATH": palace_path} if palace_path else None
    s1 = _mcp_session(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            _call("mempalace_search", {"query": "upgrade smoke-check health probe"}, 3),
        ],
        env=real_env,
        server_cmd=server_cmd,
    )

    # (1) version
    init = s1.get(1, {})
    reported = (
        init.get("result", {}).get("serverInfo", {}).get("version", "") if "result" in init else ""
    )
    if reported:
        report.checks.append(
            CheckResult(
                "version",
                version_meets(reported, min_version),
                f"server reports {reported} (floor {min_version})",
            )
        )
    else:
        report.checks.append(
            CheckResult(
                "version",
                False,
                f"no serverInfo.version in initialize response: {init or 'no response'}",
            )
        )

    # (2) advertised tools — the staleness catch
    listed = s1.get(2, {})
    tools = listed.get("result", {}).get("tools", []) if "result" in listed else []
    advertised = [t.get("name") for t in tools if isinstance(t, dict)]
    missing = missing_required_tools(advertised, EXPECTED_TOOLS)
    if not advertised:
        report.checks.append(
            CheckResult(
                "tools_advertised",
                False,
                f"tools/list returned no tools: {listed or 'no response'}",
            )
        )
    elif missing:
        report.checks.append(
            CheckResult("tools_advertised", False, f"missing advertised tools: {missing}")
        )
    else:
        report.checks.append(
            CheckResult(
                "tools_advertised",
                True,
                f"all {len(EXPECTED_TOOLS)} expected tools advertised (incl. mempalace_diary_write)",
            )
        )

    # (4) search returns without raising
    search = s1.get(3, {})
    if "error" in search:
        report.checks.append(CheckResult("search", False, f"search raised: {search['error']}"))
    elif "result" in search:
        report.checks.append(CheckResult("search", True, "search returned without error"))
    else:
        report.checks.append(
            CheckResult("search", False, f"no search response: {search or 'none'}")
        )

    # ── Session 2: diary round-trip against a DISPOSABLE sandbox palace ──
    created_sandbox = sandbox_path is None
    sandbox = sandbox_path or tempfile.mkdtemp(prefix="mempalace_smoke_sandbox_")
    token = _probe_token()
    try:
        s2 = _mcp_session(
            [
                _call(
                    "mempalace_diary_write",
                    {
                        "agent_name": _HEALTHCHECK_AGENT,
                        "entry": f"upgrade smoke-check round-trip {token}",
                        "topic": "smoke",
                    },
                    1,
                ),
                _call(
                    "mempalace_diary_read",
                    {"agent_name": _HEALTHCHECK_AGENT, "last_n": 5},
                    2,
                ),
            ],
            env={"MEMPALACE_PALACE_PATH": sandbox},
            server_cmd=server_cmd,
        )
        write_resp = s2.get(1, {})
        read_text = _result_text(s2.get(2, {}))
        if "error" in write_resp:
            report.checks.append(
                CheckResult("diary_roundtrip", False, f"diary_write raised: {write_resp['error']}")
            )
        elif token in read_text:
            report.checks.append(
                CheckResult(
                    "diary_roundtrip", True, "probe token written and read back from sandbox palace"
                )
            )
        else:
            report.checks.append(
                CheckResult(
                    "diary_roundtrip",
                    False,
                    f"probe token not found on read-back (write={write_resp.get('result', write_resp)})",
                )
            )
    finally:
        if created_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MemPalace live-MCP upgrade smoke-check (post-restart gate)."
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Configured palace path to probe read-only. Default: server's own config/env.",
    )
    parser.add_argument(
        "--sandbox",
        default=None,
        help="Disposable palace for the diary round-trip. Default: auto temp dir (auto-removed).",
    )
    parser.add_argument(
        "--min-version", default=MIN_VERSION, help="Minimum acceptable server version."
    )
    args = parser.parse_args(argv)

    report = run_smoke(
        palace_path=args.palace,
        sandbox_path=args.sandbox,
        min_version=args.min_version,
    )
    print(report.format())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
