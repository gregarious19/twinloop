"""FastAPI server (SPEC §6.5). Local only: the host publishes 127.0.0.1:8000."""

import json
import os
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.jobs import JobManager
from twinloop import segmentation
from twinloop.casefiles import initial_status, new_case_id, write_json_atomic
from twinloop.io import UNSUPPORTED

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 1024**3
CHUNK = 1024 * 1024
CASE_ID_RE = re.compile(r"^c\d{14}_[0-9a-f]{6}$")


def _image_ext(filename: str) -> str | None:
    name = filename.lower()
    for ext in ("nii.gz", "nii", "zip"):
        if name.endswith("." + ext):
            return ext
    return None


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    size = 0
    with open(dest, "wb") as out:
        while chunk := await upload.read(CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File larger than 1 GB")
            out.write(chunk)


def create_app() -> FastAPI:
    data_root = Path(os.environ.get("TWINLOOP_DATA", "/data"))
    cases_root = data_root / "runs" / "cases"
    manager = JobManager(cases_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        manager.recover()
        yield
        manager.stop()

    app = FastAPI(title="TwinLoop", lifespan=lifespan)

    def case_dir_or_404(case_id: str) -> Path:
        if not CASE_ID_RE.match(case_id) or not (cases_root / case_id / "status.json").is_file():
            raise HTTPException(404, "Case not found")
        return cases_root / case_id

    @app.get("/api/health")
    def health():
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
        return {
            "status": "ok",
            "cuda": cuda,
            "model_available": segmentation.model_available(),
            "queue_length": manager.queue_length(),
        }

    @app.post("/api/cases", status_code=202)
    async def create_case(
        image: UploadFile,
        mask: UploadFile | None = None,
        aortic_pressure_mmhg: float = Form(90.0),
    ):
        ext = _image_ext(image.filename or "")
        if ext is None:
            raise HTTPException(400, UNSUPPORTED)
        mask_ext = None
        if mask is not None and mask.filename:
            mask_ext = _image_ext(mask.filename)
            if mask_ext not in ("nii.gz", "nii"):
                raise HTTPException(400, UNSUPPORTED)
        if not 60.0 <= aortic_pressure_mmhg <= 140.0:
            raise HTTPException(400, "aortic_pressure_mmhg must be between 60 and 140")

        case_id = new_case_id()
        case_dir = cases_root / case_id
        case_dir.mkdir(parents=True)
        try:
            await _save_upload(image, case_dir / f"input.{ext}")
            if mask_ext:
                await _save_upload(mask, case_dir / f"input_mask.{mask_ext}")
        except HTTPException:
            shutil.rmtree(case_dir, ignore_errors=True)
            raise
        write_json_atomic(
            case_dir / "status.json",
            initial_status(case_id, image.filename, mask_ext is not None, aortic_pressure_mmhg),
        )
        manager.submit(case_dir)
        return {"case_id": case_id, "status": "queued"}

    @app.get("/api/cases")
    def list_cases():
        statuses = []
        for status_file in cases_root.glob("*/status.json"):
            try:
                statuses.append(json.loads(status_file.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(statuses, key=lambda s: s["case_id"], reverse=True)

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        return json.loads((case_dir_or_404(case_id) / "status.json").read_text())

    @app.get("/api/cases/{case_id}/result")
    def get_result(case_id: str):
        case_dir = case_dir_or_404(case_id)
        result = case_dir / "result.json"
        status = json.loads((case_dir / "status.json").read_text())["status"]
        if status != "done" or not result.is_file():
            return JSONResponse({"detail": "Result not ready"}, status_code=409)
        return Response(result.read_text(), media_type="application/json")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
