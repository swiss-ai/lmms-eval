#!/usr/bin/env python3
"""Unified dependency installer for lmms-eval.

Handles base package installation, model-specific extras, task-specific
extras, and runtime data (Java, NLTK, spacy) in a single script.

Usage from SLURM scripts:
    python examples/install_deps.py \\
        --model chameleon \\
        --tasks "coco_cap,flickr30k" \\
        --eval-dir /path/to/lmms-eval

Usage from Python:
    from examples.install_deps import install_all
    result = install_all(model="chameleon", tasks="coco_cap", eval_dir=".")
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

log = logging.getLogger("install_deps")

# ------------------------------------------------------------------
# Model -> pyproject.toml extras mapping
# ------------------------------------------------------------------
MODEL_TO_EXTRAS: dict[str, list[str]] = {
    # Original EMU models (older transformers pins)
    "emu3": ["emu3"],
    "llama_emu3": ["emu3"],
    "emu3p5": ["emu3_5"],
    "llama_emu3p5": ["emu3_5"],
    "llama_emu3p5_simple": ["emu3_5"],
    # Apertus models (transformers>=4.56,<5)
    "apertus_emu3": ["apertus_emu3"],
    "apertus_emu3_simple": ["apertus_emu3"],
    "apertus_emu3p5": ["apertus_emu3p5"],
    "apertus_emu3p5_simple": ["apertus_emu3p5"],
    # Baseline models
    "chameleon": ["chameleon"],
    "llava_onevision1_5": ["llava_onevision1_5"],
    "llava_onevision1_5_chat": ["llava_onevision1_5"],
    # llama_vision: no extra needed (works with base deps)
}

# Task suffixes stripped iteratively when matching extras
_TASK_SUFFIXES = (
    "_test",
    "_val",
    "_lite",
    "_pro",
    "_cot",
    "_solution",
    "_testmini",
    "_polos",
    "_train",
)

# Tasks that need runtime data (Java for METEOR / NLTK)
_METEOR_TASKS = frozenset(
    [
        "coco2014_cap",
        "coco2017_cap",
        "coco_karpathy",
        "coco_cap",
        "flickr30k",
        "nocaps",
        "textcaps",
        "detailcaps",
        "youcook2",
        "vatex",
        "refcoco",
        "refcoco+",
        "refcocog",
        "screenspot",
        "funqa",
        "cuva",
    ]
)
_CHAIR_TASKS = frozenset(["coco_cap_chair"])


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------
@dataclass
class InstallResult:
    """Summary of what was installed."""

    base_installed: bool = False
    model_extras: list[str] = field(default_factory=list)
    task_extras: list[str] = field(default_factory=list)
    runtime_data: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _run_pip(
    args: list[str],
    description: str = "",
    cwd: str | pathlib.Path | None = None,
) -> bool:
    """Run a pip command via ``sys.executable -m pip``.

    Returns True on success, False on failure.
    """
    cmd = [sys.executable, "-m", "pip"] + args
    label = description or " ".join(args[:3])
    log.info("pip %s", label)
    try:
        subprocess.check_call(cmd, cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        log.warning("pip %s failed", label)
        return False


def _get_available_extras(
    eval_dir: str | pathlib.Path,
) -> set[str]:
    """Parse pyproject.toml and return optional-dependency names."""
    toml_path = pathlib.Path(eval_dir) / "pyproject.toml"
    if not toml_path.exists():
        log.warning("pyproject.toml not found at %s", toml_path)
        return set()

    if tomllib is None:
        log.warning("tomllib/tomli not available, cannot parse pyproject.toml")
        return set()

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    extras = data.get("project", {}).get("optional-dependencies", {})
    return set(extras.keys())


def _extract_base_tasks(tasks_str: str) -> list[str]:
    """Strip known suffixes from task names, deduplicate, return sorted.

    Suffixes are stripped iteratively so compound names like
    ``coco2017_cap_val_lite`` resolve to ``coco2017_cap``.
    """
    raw = [t.strip() for t in tasks_str.split(",") if t.strip()]
    bases: set[str] = set()
    for task in raw:
        name = task
        changed = True
        while changed:
            changed = False
            for suffix in _TASK_SUFFIXES:
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    changed = True
                    break
        bases.add(name)
    return sorted(bases)


def _install_extras(
    extras: list[str],
    available: set[str],
    eval_dir: str,
) -> tuple[list[str], list[str]]:
    """Install a list of pyproject.toml extras.

    Returns (installed, errors) tuples.
    """
    installed: list[str] = []
    errors: list[str] = []
    for extra in extras:
        if extra not in available:
            log.warning(
                "Extra '%s' not found in pyproject.toml, skipping",
                extra,
            )
            continue
        if _run_pip(
            ["install", "-e", f".[{extra}]"],
            f"install .[{extra}]",
            cwd=eval_dir,
        ):
            installed.append(extra)
        else:
            errors.append(f"pip install .[{extra}] failed")
    return installed, errors


def _install_java() -> bool:
    """Install Java 17 via install-jdk if not already available."""
    if shutil.which("java"):
        log.info("Java already available")
        return True

    log.info("Java not found, installing via install-jdk...")
    if not _run_pip(["install", "-q", "install-jdk"], "install install-jdk"):
        return False

    try:
        import jdk  # type: ignore[import-untyped]

        jdk_dir = jdk.install("17")
        bin_dir = pathlib.Path("/usr/local/bin")
        java_bin = pathlib.Path(jdk_dir) / "bin" / "java"
        link = bin_dir / "java"
        if java_bin.exists() and not link.exists():
            try:
                link.symlink_to(java_bin)
                log.info("Linked %s -> %s", java_bin, link)
            except PermissionError:
                log.warning(
                    "Cannot symlink java to %s (permission denied)." " Add %s/bin to PATH manually.",
                    link,
                    jdk_dir,
                )
        os.environ["JAVA_HOME"] = jdk_dir
        log.info("JAVA_HOME=%s", jdk_dir)
        return True
    except Exception as exc:
        log.warning("Failed to install Java: %s", exc)
        return False


def _install_nltk_data(packages: list[str]) -> bool:
    """Download NLTK data packages (idempotent).

    Returns True if all packages were downloaded successfully.
    """
    try:
        import nltk  # type: ignore[import-untyped]

        for pkg in packages:
            nltk.download(pkg, quiet=True)
            log.info("NLTK %s: ok", pkg)
        return True
    except Exception as exc:
        log.warning("NLTK download failed: %s", exc)
        return False


def _install_spacy_model(name: str) -> bool:
    """Download a spacy model."""
    log.info("Installing spacy model %s...", name)
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", name, "--quiet"])
        log.info("spacy %s: ok", name)
        return True
    except subprocess.CalledProcessError:
        log.warning("Failed to install spacy model %s", name)
        return False


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def fixup_environment() -> None:
    """Apply environment fixups needed inside NGC containers."""
    # Unset SSL_CERT_FILE if it points to a non-existent path
    ssl_cert = os.environ.get("SSL_CERT_FILE", "")
    if ssl_cert and not pathlib.Path(ssl_cert).exists():
        log.info(
            "Unsetting SSL_CERT_FILE (path does not exist: %s)",
            ssl_cert,
        )
        del os.environ["SSL_CERT_FILE"]

    # Uninstall jupyterlab (conflicts with some deps)
    _run_pip(["uninstall", "jupyterlab", "-y"], "uninstall jupyterlab")


def install_model_deps(model: str, eval_dir: str) -> tuple[list[str], list[str]]:
    """Install model-specific pyproject.toml extras.

    Always includes ``metrics``.
    Returns (installed, errors) tuples.
    """
    extras_to_install: list[str] = ["metrics"]

    model_extras = MODEL_TO_EXTRAS.get(model, [])
    if model_extras:
        extras_to_install.extend(model_extras)
        log.info("Model '%s' maps to extras: %s", model, model_extras)
    else:
        log.info("Model '%s' has no extra dependencies", model)

    available = _get_available_extras(eval_dir)
    return _install_extras(extras_to_install, available, eval_dir)


def install_task_deps(tasks: str, eval_dir: str) -> tuple[list[str], list[str]]:
    """Install task-specific pyproject.toml extras.

    Strips task suffixes (_val, _test, etc.) and matches against
    available extras in pyproject.toml.
    Returns (installed, errors) tuples.
    """
    if not tasks:
        return [], []

    base_tasks = _extract_base_tasks(tasks)
    available = _get_available_extras(eval_dir)

    log.info("Base tasks extracted: %s", base_tasks)

    # Only attempt extras that match a base task name
    matching = [t for t in base_tasks if t in available]
    if not matching:
        log.info("No task-specific pip extras to install")
        return [], []

    return _install_extras(matching, available, eval_dir)


def install_runtime_data(tasks: str) -> tuple[list[str], list[str]]:
    """Install non-pip runtime data: Java, NLTK data, spacy models.

    Returns (installed, errors) tuples.
    """
    if not tasks:
        return [], []

    raw_tasks = {t.strip() for t in tasks.split(",") if t.strip()}
    base_tasks = set(_extract_base_tasks(tasks))
    all_tasks = raw_tasks | base_tasks

    installed: list[str] = []
    errors: list[str] = []

    needs_meteor = bool(all_tasks & _METEOR_TASKS)
    needs_chair = bool(all_tasks & _CHAIR_TASKS)
    needs_detailcaps = "detailcaps" in all_tasks

    if not (needs_meteor or needs_chair):
        log.info("No METEOR tasks detected, skipping runtime data")
        return installed, errors

    log.info("Installing runtime data for captioning benchmarks...")

    # Java (needed by pycocoevalcap METEOR scorer)
    if _install_java():
        installed.append("java")
    else:
        errors.append("Java installation failed")

    # NLTK wordnet (needed by METEOR scorer)
    if _install_nltk_data(["wordnet"]):
        installed.append("nltk-wordnet")
    else:
        errors.append("NLTK wordnet download failed")

    # CHAIR needs additional NLTK data
    if needs_chair:
        log.info("Installing NLTK data for CHAIR metric...")
        chair_pkgs = [
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ]
        if _install_nltk_data(chair_pkgs):
            installed.append("nltk-chair")
        else:
            errors.append("NLTK CHAIR data download failed")

    # DetailCaps needs spacy model
    if needs_detailcaps:
        if _install_spacy_model("en_core_web_sm"):
            installed.append("spacy-en_core_web_sm")
        else:
            errors.append("spacy en_core_web_sm install failed")

    return installed, errors


def install_all(
    model: str,
    tasks: str,
    eval_dir: str,
    *,
    skip_base: bool = False,
    skip_fixups: bool = False,
) -> InstallResult:
    """Orchestrate full dependency installation.

    1. Environment fixups (jupyterlab, SSL_CERT_FILE)
    2. Base package (``pip install -e .``)
    3. Model-specific extras
    4. Task-specific extras
    5. Runtime data (Java, NLTK, spacy)
    """
    result = InstallResult()

    # 1. Fixups
    if not skip_fixups:
        log.info("Applying environment fixups...")
        fixup_environment()

    # 2. Base install
    if not skip_base:
        log.info("Installing base lmms-eval package...")
        result.base_installed = _run_pip(
            ["install", "-e", "."],
            "install base package",
            cwd=eval_dir,
        )
        if not result.base_installed:
            result.errors.append("Base package installation failed")
    else:
        log.info("Skipping base package installation")

    # 3. Model extras (includes metrics)
    installed, errs = install_model_deps(model, eval_dir)
    result.model_extras = installed
    result.errors.extend(errs)

    # 4. Task extras
    installed, errs = install_task_deps(tasks, eval_dir)
    result.task_extras = installed
    result.errors.extend(errs)

    # 5. Runtime data
    installed, errs = install_runtime_data(tasks)
    result.runtime_data = installed
    result.errors.extend(errs)

    log.info("Installation complete")
    log.info(
        "  Base: %s | Model extras: %s | Task extras: %s" " | Runtime: %s",
        result.base_installed,
        result.model_extras,
        result.task_extras,
        result.runtime_data,
    )
    if result.errors:
        log.warning("Errors: %s", result.errors)

    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _list_models() -> None:
    """Print all model-to-extras mappings."""
    print("Model -> Extras mapping:")
    print("-" * 50)
    max_len = max(len(m) for m in MODEL_TO_EXTRAS)
    for model, extras in sorted(MODEL_TO_EXTRAS.items()):
        print(f"  {model:<{max_len}}  -> {', '.join(extras)}")

    # Derive models without extras from registered models
    # (hard to auto-detect; list known ones explicitly)
    print()
    print("Models with no extra deps (use base): llama_vision")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=("Install dependencies for lmms-eval models and tasks."),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name (e.g. 'chameleon', 'apertus_emu3')",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help=("Comma-separated task names" " (e.g. 'coco_cap,flickr30k')"),
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.getcwd(),
        help="Path to lmms-eval repo root (default: cwd)",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base package installation (pip install -e .)",
    )
    parser.add_argument(
        "--skip-fixups",
        action="store_true",
        help="Skip environment fixups (jupyterlab, SSL_CERT_FILE)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all model-to-extras mappings and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_models:
        _list_models()
        return

    if not args.model and not args.tasks:
        parser.error("At least one of --model or --tasks is required")

    result = install_all(
        model=args.model,
        tasks=args.tasks,
        eval_dir=args.eval_dir,
        skip_base=args.skip_base,
        skip_fixups=args.skip_fixups,
    )

    if result.errors:
        log.error("Errors encountered: %s", result.errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
