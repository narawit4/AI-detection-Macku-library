"""Validate and copy the metadata required beside a packaged Jitter binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parent
_NAME_SEPARATOR = re.compile(r"[-_.]+")
_PINNED_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)"
)


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


def _environment_tokens(environ: Mapping[str, str], name: str) -> list[str]:
    value = environ.get(name, "")
    tokens = value.split()
    if not tokens:
        raise ValueError(f"missing packaging environment variable: {name}")
    return tokens


def review_payload(
    *, root: Path = ROOT, environ: Mapping[str, str] = os.environ
) -> dict[str, object]:
    requirements = parse_pinned_requirements(
        (root / "requirements.txt").read_text(encoding="utf-8")
    )
    manifest_path = root / "licenses" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = validate_license_manifest(manifest, root, requirements)
    return {
        "compile_targets": _environment_tokens(
            environ, "JITTER_PACKAGE_COMPILE_TARGETS"
        ),
        "runtime_imports": _environment_tokens(
            environ, "JITTER_PACKAGE_RUNTIME_IMPORTS"
        ),
        "nuitka_data_options": _environment_tokens(
            environ, "JITTER_PACKAGE_NUITKA_DATA_OPTIONS"
        ),
        "release_materials": _environment_tokens(
            environ, "JITTER_PACKAGE_RELEASE_MATERIALS"
        ),
        "requirements": [
            {"name": requirement.name, "version": requirement.version}
            for requirement in requirements
        ],
        "licensed_packages": [
            {"name": record.name, "version": record.version} for record in records
        ],
    }


def copy_release_materials(
    output_dir: Path | str,
    *,
    root: Path = ROOT,
    materials: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    selected = tuple(
        materials
        if materials is not None
        else _environment_tokens(os.environ, "JITTER_PACKAGE_RELEASE_MATERIALS")
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--review-json", action="store_true")
    action.add_argument("--copy-release-materials", metavar="OUTPUT_DIR")
    args = parser.parse_args(argv)
    if args.review_json:
        print(json.dumps(review_payload(), sort_keys=True))
    else:
        review_payload()
        copy_release_materials(args.copy_release_materials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
