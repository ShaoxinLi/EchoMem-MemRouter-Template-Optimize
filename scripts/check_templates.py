"""Template governance checker for MemRouter.

Usage:
    python scripts/check_templates.py

Checks:
1. No disabled templates in active directory (templates_data/)
2. No duplicate active (backend_id, intent_family) pairs
3. Minimum prototype / hard-negative counts per template
4. Consistent schema fields
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

REQUIRED_TOP_KEYS = {"schema_version", "template_id", "version", "status", "target", "intent_family"}
REQUIRED_TARGET_KEYS = {"primary_backend_id"}

MIN_PROTOTYPES = 10
MIN_HARD_NEGATIVES = 5


def load_template(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_active_directory(templates_dir: Path) -> List[str]:
    errors: List[str] = []
    for path in sorted(templates_dir.glob("*.yaml")):
        data = load_template(path)
        status = data.get("status", "enabled")
        if status == "disabled":
            errors.append(f"ACTIVE_DIR_DISABLED: {path.name} has status=disabled but lives in active directory")
    return errors


def check_schema(data: Dict[str, Any], path: Path) -> List[str]:
    errors: List[str] = []
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        errors.append(f"SCHEMA_MISSING_KEYS [{path.name}]: {', '.join(sorted(missing))}")

    target = data.get("target", {})
    missing_target = REQUIRED_TARGET_KEYS - set(target.keys())
    if missing_target:
        errors.append(f"SCHEMA_MISSING_TARGET [{path.name}]: {', '.join(sorted(missing_target))}")

    return errors


def check_duplicates(templates_dir: Path) -> List[str]:
    errors: List[str] = []
    seen: Dict[Tuple[str, str], str] = {}
    for path in sorted(templates_dir.glob("*.yaml")):
        data = load_template(path)
        if data.get("status") != "enabled":
            continue
        target = data.get("target", {})
        backend_id = target.get("primary_backend_id")
        intent_family = data.get("intent_family", {}).get("name")
        if not backend_id or not intent_family:
            continue
        key = (backend_id, intent_family)
        if key in seen:
            errors.append(
                f"DUPLICATE_ACTIVE [{path.name}]: "
                f"({backend_id}, {intent_family}) already active in {seen[key]}"
            )
        else:
            seen[key] = path.name
    return errors


def check_counts(data: Dict[str, Any], path: Path) -> List[str]:
    errors: List[str] = []
    protos = data.get("query_prototypes", [])
    negs = data.get("hard_negatives", [])
    template_id = data.get("template_id", path.name)

    if len(protos) < MIN_PROTOTYPES:
        errors.append(
            f"LOW_PROTOTYPES [{template_id}]: {len(protos)} < {MIN_PROTOTYPES}"
        )
    if len(negs) < MIN_HARD_NEGATIVES:
        errors.append(
            f"LOW_HARD_NEGATIVES [{template_id}]: {len(negs)} < {MIN_HARD_NEGATIVES}"
        )
    return errors


def main() -> int:
    project_root = Path(__file__).parent.parent
    templates_dir = project_root / "echomem" / "templates_data"
    archive_dir = project_root / "echomem" / "templates_archive"

    if not templates_dir.exists():
        print(f"ERROR: templates_dir not found: {templates_dir}")
        return 1

    all_errors: List[str] = []
    active_count = 0
    disabled_count = 0

    # 1. Active directory must not contain disabled templates
    all_errors.extend(check_active_directory(templates_dir))

    # 2–5. Per-template checks
    for path in sorted(templates_dir.glob("*.yaml")):
        data = load_template(path)
        status = data.get("status", "enabled")
        if status == "enabled":
            active_count += 1
        else:
            disabled_count += 1
            continue

        all_errors.extend(check_schema(data, path))
        all_errors.extend(check_counts(data, path))

    # Duplicate check across active templates
    all_errors.extend(check_duplicates(templates_dir))

    # Archive directory check
    archived = list(archive_dir.glob("*.yaml")) if archive_dir.exists() else []

    # Summary
    print("=" * 60)
    print("Template Governance Report")
    print("=" * 60)
    print(f"Active templates:   {active_count}")
    print(f"Disabled templates: {disabled_count}")
    print(f"Archived templates: {len(archived)}")
    print(f"Errors found:       {len(all_errors)}")
    print("=" * 60)

    if all_errors:
        for err in all_errors:
            print(f"  - {err}")
        print("\nFAILED")
        return 1
    else:
        print("All checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
