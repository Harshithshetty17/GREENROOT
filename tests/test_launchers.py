"""The one-click launchers.

A launcher that skips the dependency install when it should not have is a
worse failure than no launcher at all: the app starts, gets partway through
its imports, and drops the person into a Python traceback on a screen that
was supposed to be a crop recommendation. That is exactly what happened when
``bcrypt`` was added to requirements.txt and the launchers' hand-written
probe list was not updated with it -- the probe passed, the install was
skipped, and ``run.bat`` failed at ``import bcrypt``.

So the probe is checked against the imports the code actually makes, rather
than trusted to be maintained by hand.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("run.bat", "run_phone.bat", "run_phone.sh")

#: Present in requirements.txt but imported only on a path the app can reach
#: without them -- an .xlsx upload, icon generation. Their absence does not
#: stop the app from starting, so the probe need not demand them.
LAZY = frozenset({"openpyxl", "PIL", "pillow"})


def third_party_imports() -> set:
    """Every non-stdlib top-level module imported by the app and its package."""
    stdlib = set(sys.stdlib_module_names)
    found: set = set()
    for path in [ROOT / "app.py"] + sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found
            if m not in stdlib and m != "src" and not m.startswith("_")}


def probe_of(launcher: str) -> set:
    """The module list the launcher tests before deciding to skip the install."""
    text = (ROOT / launcher).read_text(encoding="utf-8")
    match = re.search(r'-c "import ([^"]+)"', text)
    assert match, f"{launcher} has no recognisable dependency probe"
    return {m.strip() for m in match.group(1).split(",")}


@pytest.mark.parametrize("launcher", LAUNCHERS)
class TestDependencyProbe:
    def test_the_launcher_exists(self, launcher):
        assert (ROOT / launcher).is_file()

    def test_it_probes_every_import_the_app_needs_to_start(self, launcher):
        missing = third_party_imports() - probe_of(launcher) - LAZY
        assert not missing, (
            f"{launcher} would skip the install while these are absent: "
            + ", ".join(sorted(missing))
        )

    def test_it_probes_nothing_the_app_does_not_import(self, launcher):
        """A stale name in the probe fails the check forever and reinstalls
        on every launch, which is its own kind of broken."""
        extra = probe_of(launcher) - third_party_imports()
        assert not extra, f"{launcher} probes unused modules: {sorted(extra)}"

    def test_bcrypt_specifically(self, launcher):
        """Named for the regression that prompted this file: accounts were
        added, requirements.txt was updated, the probes were not, and the
        first person to run it hit ModuleNotFoundError."""
        assert "bcrypt" in probe_of(launcher)


class TestProbeMatchesRequirements:
    def test_every_probed_module_is_a_declared_dependency(self):
        """Probing something pip will not install is a launcher that can
        never satisfy itself."""
        declared = set()
        for line in (ROOT / "requirements.txt").read_text(
                encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                declared.add(re.split(r"[<>=!~\[]", line)[0].strip().lower())
        # Import name -> distribution name, where they differ.
        aliases = {"sklearn": "scikit-learn", "fpdf": "fpdf2",
                   "PIL": "pillow", "lime": "lime", "shap": "shap"}
        for module in probe_of("run.bat"):
            name = aliases.get(module, module).lower()
            assert name in declared or module.lower() in declared, (
                f"{module} is probed but not in requirements.txt"
            )
