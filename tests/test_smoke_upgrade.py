"""
test_smoke_upgrade.py — TDD assertions for the live-MCP upgrade smoke-check (A1).

The smoke-check is the post-restart gate codifying this session's #1 lesson:
diary_write was present in the code but a stale running server failed to
*advertise* it to the client. The gate connects to a freshly-launched server
over the real stdio JSON-RPC wire (NOT a module import) and asserts:

  1. serverInfo.version >= 3.4.0
  2. tools/list advertises every expected tool — load-bearing: mempalace_diary_write
  3. diary_write -> diary_read round-trips (against a DISPOSABLE sandbox palace)
  4. search returns without raising

Design guardrails exercised by these tests:
  - version + advertise + search run read-only against the configured palace.
  - the diary round-trip writes ONLY to a throwaway sandbox palace
    (MEMPALACE_PALACE_PATH=tmpdir); the configured palace gains no drawer.
"""

import subprocess

import pytest

import mempalace.smoke_upgrade as su
from mempalace.smoke_upgrade import (
    EXPECTED_TOOLS,
    SmokeReport,
    main,
    missing_required_tools,
    run_smoke,
    version_meets,
)


# ── Pure logic: version gate ─────────────────────────────────────────────


def test_version_meets_equal_passes():
    assert version_meets("3.4.0", "3.4.0") is True


@pytest.mark.parametrize("reported", ["3.4.1", "3.5.0", "3.10.0", "4.0.0"])
def test_version_meets_higher_passes(reported):
    assert version_meets(reported, "3.4.0") is True


@pytest.mark.parametrize("reported", ["3.3.9", "3.3.0", "2.9.9", "3.0.0"])
def test_version_meets_lower_fails(reported):
    assert version_meets(reported, "3.4.0") is False


def test_version_meets_two_component_is_padded():
    # "3.4" must be treated as 3.4.0 — meets the 3.4.0 floor.
    assert version_meets("3.4", "3.4.0") is True
    # "3.3" is below.
    assert version_meets("3.3", "3.4.0") is False


def test_version_parse_prerelease_suffix_is_not_mis_read_as_patch():
    # Polish note (3): a pre-release like "3.4.0rc1" must parse the base
    # version, NOT read "0rc1" as patch 1. Leading-digits-per-component.
    assert su._parse_version("3.4.0rc1") == [3, 4, 0]
    assert version_meets("3.4.0rc1", "3.4.0") is True  # rc of 3.4.0 satisfies the floor
    assert version_meets("3.3.0rc5", "3.4.0") is False  # 3.3.x rc is still below


def test_mcp_session_timeout_yields_empty_not_crash(monkeypatch):
    # Polish note (2): a hung server must produce a clean FAIL report, not crash
    # the gate. _mcp_session catches subprocess.TimeoutExpired and returns {}.
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="server", timeout=1)

    monkeypatch.setattr(su.subprocess, "run", _raise_timeout)
    responses = su._mcp_session([{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    assert responses == {}  # no responses, no exception


def test_run_smoke_reports_fail_when_server_hangs(monkeypatch):
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="server", timeout=1)

    monkeypatch.setattr(su.subprocess, "run", _raise_timeout)
    report = run_smoke(palace_path="/tmp/does-not-matter")
    assert isinstance(report, SmokeReport)
    assert report.ok is False  # FAIL, not a crash


# ── Pure logic: advertised-tool gate (the staleness catch) ───────────────


def test_expected_tools_includes_diary_write():
    # The one that silently disappeared. If it ever leaves the expected set,
    # the gate stops protecting against the exact failure it exists for.
    assert "mempalace_diary_write" in EXPECTED_TOOLS


def test_missing_required_tools_flags_absent_diary_write():
    advertised = [t for t in EXPECTED_TOOLS if t != "mempalace_diary_write"]
    missing = missing_required_tools(advertised, EXPECTED_TOOLS)
    assert "mempalace_diary_write" in missing


def test_missing_required_tools_empty_when_all_present():
    assert missing_required_tools(list(EXPECTED_TOOLS), EXPECTED_TOOLS) == []


# ── Integration: real stdio JSON-RPC wire against a freshly-launched server ─


def test_live_server_advertises_diary_write_over_the_wire(palace_path):
    """tools/list over the real wire must include mempalace_diary_write.
    This is the assertion that would have caught the stale-server bug."""
    report = run_smoke(palace_path=palace_path)
    advertise = _by_name(report, "tools_advertised")
    assert advertise is not None, "smoke report missing the advertise check"
    assert advertise.passed, f"advertise check failed: {advertise.detail}"


def test_live_server_reports_version_at_or_above_floor(palace_path):
    report = run_smoke(palace_path=palace_path)
    ver = _by_name(report, "version")
    assert ver is not None and ver.passed, (
        f"version check failed: {ver.detail if ver else 'absent'}"
    )
    assert "3.4.0" in ver.detail or "3.4" in ver.detail


def test_diary_roundtrip_in_sandbox_leaves_configured_palace_untouched(palace_path, tmp_path):
    """The round-trip proves diary_write functions, but must write ONLY to the
    disposable sandbox palace — the configured palace must gain no drawer."""
    sandbox = tmp_path / "sandbox_palace"
    before = _drawer_count(palace_path)

    report = run_smoke(palace_path=palace_path, sandbox_path=str(sandbox))

    roundtrip = _by_name(report, "diary_roundtrip")
    assert roundtrip is not None and roundtrip.passed, (
        f"diary round-trip failed: {roundtrip.detail if roundtrip else 'absent'}"
    )
    after = _drawer_count(palace_path)
    assert after == before, (
        f"smoke-check left a stray drawer in the configured palace: {before} -> {after}"
    )


def test_search_does_not_raise(palace_path):
    report = run_smoke(palace_path=palace_path)
    search = _by_name(report, "search")
    assert search is not None and search.passed, (
        f"search check failed: {search.detail if search else 'absent'}"
    )


def test_run_smoke_overall_ok_against_current_server(palace_path):
    report = run_smoke(palace_path=palace_path)
    assert isinstance(report, SmokeReport)
    assert report.ok, (
        f"smoke report not ok: {[(c.name, c.detail) for c in report.checks if not c.passed]}"
    )


def test_main_returns_zero_on_pass(palace_path, capsys):
    rc = main(["--palace", str(palace_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


# ── helpers ──────────────────────────────────────────────────────────────


def _by_name(report, name):
    for c in report.checks:
        if c.name == name:
            return c
    return None


def _drawer_count(palace_path):
    """Count drawers in the configured palace via the searcher's collection,
    read-only. Returns 0 when the palace was never created."""
    from mempalace.smoke_upgrade import drawer_count

    return drawer_count(str(palace_path))
