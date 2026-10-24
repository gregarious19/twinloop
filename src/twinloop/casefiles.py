"""Case ids and status.json helpers shared by the CLI and the job runner (SPEC §6.4)."""

import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_case_id() -> str:
    return "c" + datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "_" + secrets.token_hex(3)


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def initial_status(
    case_id: str, filename: str, has_mask: bool, aortic_pressure_mmhg: float, status: str = "queued"
) -> dict:
    now = utc_now_iso()
    return {
        "case_id": case_id,
        "status": status,
        "filename": filename,
        "created_at": now,
        "updated_at": now,
        "has_mask": has_mask,
        "aortic_pressure_mmhg": aortic_pressure_mmhg,
        "error": None,
        "timings_s": {},
    }


def read_status(case_dir: Path) -> dict:
    return json.loads((Path(case_dir) / "status.json").read_text())


def update_status(case_dir: Path, **changes) -> dict:
    """Merge `changes` into status.json (atomically) and refresh `updated_at`."""
    status = read_status(case_dir)
    status.update(changes)
    status["updated_at"] = utc_now_iso()
    write_json_atomic(Path(case_dir) / "status.json", status)
    return status
