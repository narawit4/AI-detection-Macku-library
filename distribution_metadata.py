"""Validate and copy the metadata required beside a packaged Jitter binary."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, TextIO


ROOT = Path(__file__).resolve().parent
_NAME_SEPARATOR = re.compile(r"[-_.]+")
_PINNED_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)"
)
_MODEL_SHA256 = "6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C"
_DIRECTML_DLL_SHA256 = (
    "B73972115320E906A49602F2027A3266622881B0D325BA685E0F165A9482A8D7"
)
_NUITKA_PACKAGE_CONFIG = "nuitka-package.config.yml"
_NUITKA_PACKAGE_CONFIG_SHA256 = (
    "E2D715C37C2EF10D3195F1DC05997F322E9E2F136755D9860664F10E4A48D2DE"
)
_AI_RUNTIME_SELF_CHECK_ARGUMENT = "--ai-runtime-self-check"
_RELEASE_MATERIALS = ("LICENSE", "THIRD_PARTY_NOTICES.md", "licenses")
_EXCLUDED_SOURCE_PARTS = {
    ".git", ".superpowers", ".worktrees", "__pycache__", "build-output",
    "dist", "tests",
}
_PROHIBITED_SOURCE_TOKENS = {
    "training", "profile", "profiles", "tray", "ai_tracker"
}


@dataclass(frozen=True)
class PinnedRequirement:
    name: str
    version: str


@dataclass(frozen=True)
class LicenseRecord:
    name: str
    version: str
    license_expression: str
    source_required: bool
    source_archive: Path | None


@dataclass(frozen=True)
class RuntimeImportRecord:
    import_root: str
    distribution: str
    required_by: str
    transitive_of: str | None = None


@dataclass(frozen=True)
class NuitkaPackageConfiguration:
    path: str
    config_sha256: str
    module: str
    source: str
    destination: str
    sha256: str

    def to_payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "config_sha256": self.config_sha256,
            "module": self.module,
            "source": self.source,
            "destination": self.destination,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BuildPlan:
    """The complete, reviewed command and release plan for one packaged build."""

    root: Path
    compile_targets: tuple[str, ...]
    runtime_imports: tuple[str, ...]
    nuitka_data_options: tuple[str, ...]
    release_materials: tuple[str, ...]
    requirements: tuple[PinnedRequirement, ...]
    licensed_packages: tuple[LicenseRecord, ...]
    runtime_inventory: tuple[RuntimeImportRecord, ...]
    nuitka_package_configuration: NuitkaPackageConfiguration
    install_argv: tuple[str, ...]
    compile_argv: tuple[str, ...]
    test_argv: tuple[str, ...]
    runtime_import_argv: tuple[str, ...]
    nuitka_argv: tuple[str, ...]
    packaged_self_check_argv: tuple[str, ...]
    output_dir: Path
    build_log: Path

    def to_payload(self) -> dict[str, object]:
        return {
            "compile_targets": list(self.compile_targets),
            "runtime_imports": list(self.runtime_imports),
            "nuitka_data_options": list(self.nuitka_data_options),
            "release_materials": list(self.release_materials),
            "requirements": [
                {"name": item.name, "version": item.version}
                for item in self.requirements
            ],
            "licensed_packages": [
                {"name": item.name, "version": item.version}
                for item in self.licensed_packages
            ],
            "runtime_inventory": [
                {
                    "import_root": item.import_root,
                    "distribution": item.distribution,
                    "required_by": item.required_by,
                    **(
                        {"transitive_of": item.transitive_of}
                        if item.transitive_of is not None else {}
                    ),
                }
                for item in self.runtime_inventory
            ],
            "nuitka_package_configuration": (
                self.nuitka_package_configuration.to_payload()
            ),
            "commands": {
                "install": list(self.install_argv),
                "compile": list(self.compile_argv),
                "test": list(self.test_argv),
                "runtime_import": list(self.runtime_import_argv),
                "nuitka": list(self.nuitka_argv),
                "packaged_self_check": list(self.packaged_self_check_argv),
            },
            "output_dir": str(self.output_dir.relative_to(self.root)).replace("/", "\\"),
            "build_log": str(self.build_log.relative_to(self.root)).replace("/", "\\"),
        }


def normalize_distribution_name(name: str) -> str:
    normalized = _NAME_SEPARATOR.sub("-", name.strip()).lower()
    if not normalized:
        raise ValueError("distribution name is empty")
    return normalized


def parse_pinned_requirements(text: str) -> tuple[PinnedRequirement, ...]:
    requirements = []
    seen = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PINNED_REQUIREMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"requirement line {line_number} is not an exact pin")
        name = normalize_distribution_name(match.group("name"))
        if name in seen:
            raise ValueError(f"duplicate requirement: {name}")
        seen.add(name)
        requirements.append(PinnedRequirement(name, match.group("version")))
    return tuple(requirements)


def _artifact_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a non-empty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"artifact path escapes the source tree: {relative}")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*pure.parts).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes the source tree: {relative}")
    return resolved


def _validate_nuitka_package_configuration(
    root: Path,
) -> NuitkaPackageConfiguration:
    config_path = root / _NUITKA_PACKAGE_CONFIG
    if not config_path.is_file():
        raise ValueError(
            f"Nuitka package configuration is missing: {config_path}"
        )
    try:
        normalized_bytes = config_path.read_text(encoding="utf-8").encode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"Nuitka package configuration is unreadable: {config_path}"
        ) from error
    actual_hash = hashlib.sha256(normalized_bytes).hexdigest().upper()
    if actual_hash != _NUITKA_PACKAGE_CONFIG_SHA256:
        raise ValueError(
            f"Nuitka package configuration hash mismatch: {config_path}"
        )
    return NuitkaPackageConfiguration(
        path=_NUITKA_PACKAGE_CONFIG,
        config_sha256=_NUITKA_PACKAGE_CONFIG_SHA256,
        module="jitter_app.ai.detection",
        source="onnxruntime/capi/DirectML.dll",
        destination="onnxruntime/capi/DirectML.dll",
        sha256=_DIRECTML_DLL_SHA256,
    )


def _verify_artifact(root: Path, artifact: object, *, label: str) -> Path:
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} must be an object")
    path = _artifact_path(root, artifact.get("path"))
    expected_hash = artifact.get("sha256")
    provenance = artifact.get("provenance")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", expected_hash):
        raise ValueError(f"{label} has an invalid SHA-256")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(f"{label} has no provenance")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash.casefold() != expected_hash.casefold():
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def validate_license_manifest(
    manifest: object,
    root: Path,
    requirements: Iterable[PinnedRequirement],
) -> tuple[LicenseRecord, ...]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("license manifest schema_version must be 1")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("license manifest packages must be a non-empty list")

    records = []
    versions = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("license manifest package must be an object")
        name = normalize_distribution_name(package.get("name", ""))
        version = package.get("version")
        license_expression = package.get("license")
        if name in versions:
            raise ValueError(f"duplicate license package: {name}")
        if not isinstance(version, str) or not version:
            raise ValueError(f"license package {name} has no version")
        for field in ("copyright", "homepage", "metadata_provenance"):
            if not isinstance(package.get(field), str) or not package[field].strip():
                raise ValueError(f"license package {name} has no {field}")
        if not isinstance(license_expression, str) or not license_expression:
            raise ValueError(f"license package {name} has no license")
        artifacts = package.get("license_files")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"license package {name} has no license files")
        for index, artifact in enumerate(artifacts):
            _verify_artifact(root, artifact, label=f"{name} license file {index}")

        source = package.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            raise ValueError(f"license package {name} has no source URL")
        source_required = source.get("required_with_binary")
        if not isinstance(source_required, bool):
            raise ValueError(f"license package {name} has invalid source policy")
        archive = source.get("archive")
        source_archive = None
        if archive is not None:
            source_archive = _verify_artifact(
                root, archive, label=f"{name} source archive"
            )
        if source_required and source_archive is None:
            raise ValueError(f"license package {name} requires a source archive")

        versions[name] = version
        records.append(
            LicenseRecord(
                name, version, license_expression, source_required, source_archive
            )
        )

    for requirement in requirements:
        if versions.get(requirement.name) != requirement.version:
            raise ValueError(
                f"license manifest does not match {requirement.name}=={requirement.version}"
            )
    return tuple(records)


def validate_runtime_inventory(
    inventory: object,
    requirements: Iterable[PinnedRequirement],
    package_versions: Mapping[str, str],
    runtime_imports: Iterable[str],
) -> tuple[RuntimeImportRecord, ...]:
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("runtime inventory must be a non-empty list")
    requirement_versions = {
        requirement.name: requirement.version for requirement in requirements
    }
    by_root = {}
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("runtime inventory item must be an object")
        import_root = item.get("import_root")
        distribution = item.get("distribution")
        required_by = item.get("required_by")
        transitive_of = item.get("transitive_of")
        if not isinstance(import_root, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", import_root
        ):
            raise ValueError("runtime inventory has an invalid import root")
        if import_root in by_root:
            raise ValueError(f"duplicate runtime import root: {import_root}")
        if not isinstance(distribution, str) or not distribution.strip():
            raise ValueError(f"runtime import {import_root} has no distribution")
        if not isinstance(required_by, str) or not required_by.strip():
            raise ValueError(f"runtime import {import_root} has no required_by")
        if transitive_of is not None and (
            not isinstance(transitive_of, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", transitive_of)
        ):
            raise ValueError(f"runtime import {import_root} has invalid transitive_of")
        by_root[import_root] = RuntimeImportRecord(
            import_root,
            normalize_distribution_name(distribution),
            required_by,
            transitive_of,
        )

    selected = []
    seen_roots = set()
    for import_root in runtime_imports:
        if import_root in seen_roots:
            raise ValueError(f"duplicate packaged runtime import: {import_root}")
        seen_roots.add(import_root)
        record = by_root.get(import_root)
        if record is None:
            raise ValueError(f"runtime import has no inventory record: {import_root}")
        pinned_version = requirement_versions.get(record.distribution)
        licensed_version = package_versions.get(record.distribution)
        if pinned_version is None:
            raise ValueError(
                f"runtime import {import_root} distribution is not pinned: "
                f"{record.distribution}"
            )
        if licensed_version != pinned_version:
            raise ValueError(
                f"runtime import {import_root} distribution is not licensed at "
                f"{record.distribution}=={pinned_version}"
            )
        selected.append(record)
    return tuple(selected)


def discover_application_sources(root: Path) -> tuple[Path, ...]:
    """Return live application sources, including nested packages, in stable order."""
    sources = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if _EXCLUDED_SOURCE_PARTS.intersection(relative.parts):
            continue
        if any(part.endswith((".build", ".dist")) for part in relative.parts):
            continue
        tokens = {
            token
            for part in (*relative.parts, path.stem)
            for token in _NAME_SEPARATOR.split(part.casefold())
        }
        prohibited = tokens.intersection(_PROHIBITED_SOURCE_TOKENS)
        if prohibited:
            raise ValueError(
                f"prohibited application source path {relative}: {sorted(prohibited)}"
            )
        sources.append(path)
    if not sources:
        raise ValueError("no application Python sources found")
    return tuple(sorted(sources, key=lambda item: item.relative_to(root).as_posix()))


def _source_import_roots(root: Path, sources: Iterable[Path]) -> set[str]:
    local_roots = {
        path.relative_to(root).parts[0].removesuffix(".py") for path in sources
    }
    imported = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                imported.add(node.module.split(".", 1)[0])
    return imported.difference(sys.stdlib_module_names, local_roots)


def _active_runtime_roots(inventory: object, direct_roots: set[str]) -> tuple[str, ...]:
    if not isinstance(inventory, list):
        raise ValueError("runtime inventory must be a list")
    items_by_root = {}
    for item in inventory:
        if not isinstance(item, dict) or not isinstance(item.get("import_root"), str):
            raise ValueError("runtime inventory item has no import root")
        items_by_root[item["import_root"]] = item
    unknown = sorted(direct_roots.difference(items_by_root))
    if unknown:
        raise ValueError(f"source imports have no runtime inventory mapping: {unknown}")

    active = set(direct_roots)
    changed = True
    while changed:
        changed = False
        for import_root, item in items_by_root.items():
            if item.get("transitive_of") in active and import_root not in active:
                active.add(import_root)
                changed = True
    return tuple(
        item["import_root"] for item in inventory if item["import_root"] in active
    )


def build_plan(root: Path = ROOT) -> BuildPlan:
    """Validate inputs and construct the sole command plan used for packaging."""
    root = root.resolve()
    nuitka_package_configuration = _validate_nuitka_package_configuration(root)
    requirements = parse_pinned_requirements(
        (root / "requirements.txt").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "licenses" / "manifest.json").read_text(encoding="utf-8")
    )
    licensed_packages = validate_license_manifest(manifest, root, requirements)
    sources = discover_application_sources(root)
    direct_roots = _source_import_roots(root, sources)
    active_roots = _active_runtime_roots(
        manifest.get("runtime_inventory"), direct_roots
    )
    runtime_inventory = validate_runtime_inventory(
        manifest.get("runtime_inventory"), requirements,
        {item.name: item.version for item in licensed_packages}, active_roots,
    )

    pinned_names = {item.name for item in requirements}
    active_distributions = {item.distribution for item in runtime_inventory}
    if active_distributions != pinned_names:
        missing = sorted(active_distributions - pinned_names)
        unused = sorted(pinned_names - active_distributions)
        raise ValueError(
            "runtime dependency inventory does not exactly match pins; "
            f"missing pins={missing}, unused pins={unused}"
        )

    model_path = root / "models" / "all_games_320.onnx"
    if not model_path.is_file():
        raise ValueError(f"packaged model is missing: {model_path}")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest().upper()
    if model_hash != _MODEL_SHA256:
        raise ValueError(f"packaged model hash mismatch: {model_path}")

    sound_source = root / "jitter_app" / "presentation" / "sound.py"
    sound_assets = root / "sound"
    if sound_source.is_file() != sound_assets.is_dir():
        raise ValueError("sound source and sound assets must be packaged together")

    compile_targets = tuple(path.relative_to(root).as_posix() for path in sources)
    data_options = [
        "--include-data-files="
        "models/all_games_320.onnx=models/all_games_320.onnx",
        "--include-data-dir=licenses=licenses",
    ]
    if sound_source.is_file():
        data_options.append("--include-data-dir=sound=sound")

    python = sys.executable
    install_argv = (
        python, "-m", "pip", "install", "-r", "requirements.txt",
        "Nuitka", "ordered-set", "zstandard",
    )
    compile_argv = (python, "-m", "py_compile", *compile_targets)
    test_argv = (python, "-m", "unittest", "discover", "-s", "tests", "-v")
    runtime_import_argv = (
        python, "-c", "import " + ", ".join(active_roots)
    )
    package_config_option = (
        "--user-package-configuration-file="
        + nuitka_package_configuration.path
    )
    nuitka_argv = (
        python, "-m", "nuitka", "--onefile", "--mingw64",
        "--assume-yes-for-downloads", "--progress-bar=none",
        "--windows-console-mode=attach", "--enable-plugin=tk-inter",
        package_config_option,
        *data_options, "--output-filename=Jitter.exe",
        "--output-dir=build-output", "main.py",
    )
    output_dir = root / "build-output"
    packaged_self_check_argv = (
        str(output_dir / "Jitter.exe"),
        _AI_RUNTIME_SELF_CHECK_ARGUMENT,
    )
    return BuildPlan(
        root=root,
        compile_targets=compile_targets,
        runtime_imports=active_roots,
        nuitka_data_options=tuple(data_options),
        release_materials=_RELEASE_MATERIALS,
        requirements=requirements,
        licensed_packages=licensed_packages,
        runtime_inventory=runtime_inventory,
        nuitka_package_configuration=nuitka_package_configuration,
        install_argv=install_argv,
        compile_argv=compile_argv,
        test_argv=test_argv,
        runtime_import_argv=runtime_import_argv,
        nuitka_argv=nuitka_argv,
        packaged_self_check_argv=packaged_self_check_argv,
        output_dir=output_dir,
        build_log=output_dir / "build.log",
    )


def review_payload(*, root: Path = ROOT) -> dict[str, object]:
    return build_plan(root).to_payload()


def copy_release_materials(
    output_dir: Path | str,
    *,
    root: Path = ROOT,
    materials: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    selected = tuple(materials if materials is not None else _RELEASE_MATERIALS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative in selected:
        source = _artifact_path(root, relative)
        if not source.exists():
            raise FileNotFoundError(source)
        destination = output / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)


def execute_build_plan(
    plan: BuildPlan,
    *,
    runner: Callable[..., object] = subprocess.run,
    release_copier: Callable[..., tuple[Path, ...]] = copy_release_materials,
) -> None:
    """Execute exactly the command plan already exposed by review JSON."""
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    for argv in (
        plan.install_argv,
        plan.compile_argv,
        plan.test_argv,
        plan.runtime_import_argv,
    ):
        runner(argv, cwd=plan.root, check=True, env=None)
    with plan.build_log.open("w", encoding="utf-8") as log:
        runner(
            plan.nuitka_argv,
            cwd=plan.root,
            check=True,
            env=None,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    runner(
        plan.packaged_self_check_argv,
        cwd=plan.root,
        check=True,
        env=None,
    )
    release_copier(
        plan.output_dir, root=plan.root, materials=plan.release_materials
    )


def confirm_build(
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    plan_factory: Callable[[], BuildPlan] = build_plan,
    executor: Callable[[BuildPlan], None] = execute_build_plan,
) -> int:
    """Execute a build only after an exact, interactive confirmation."""
    print(
        "This explicitly installs packaging tools and builds "
        "build-output\\Jitter.exe.",
        file=output,
    )
    print("Type BUILD to continue: ", end="", file=output, flush=True)
    try:
        response = input_fn("")
    except (EOFError, KeyboardInterrupt):
        response = None
        print(file=output)
    if response != "BUILD":
        print("Build cancelled.", file=output)
        return 2

    plan = plan_factory()
    executor(plan)
    print("Build complete: build-output\\Jitter.exe", file=output)
    return 0


_HELP = """\
Usage: python distribution_metadata.py MODE

Validate, review, or explicitly execute Jitter's canonical packaging plan.

Modes (choose exactly one):
  --help         Show this help without starting a build.
  --review-json  Print the validated command and release plan as JSON.
  --build        Build build-output\\Jitter.exe without an extra prompt.

For the confirmed compatibility entry, run .\\gen.bat with no arguments.
"""


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--help"]:
        print(_HELP, end="")
        return 0
    if arguments == ["--confirm-build"]:
        return confirm_build()
    if arguments not in (["--review-json"], ["--build"]):
        print(
            "Invalid arguments. Use exactly --help, --review-json, or --build.",
            file=sys.stderr,
        )
        return 2

    plan = build_plan()
    if arguments == ["--review-json"]:
        print(json.dumps(plan.to_payload(), sort_keys=True))
    else:
        execute_build_plan(plan)
        print("Build complete: build-output\\Jitter.exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
