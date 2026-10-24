"""Single-worker job runner (SPEC §6.2, §6.4): one case at a time."""

import dataclasses
import queue
import threading
import time
import traceback
from pathlib import Path

from twinloop.casefiles import read_status, update_status
from twinloop.config import PipelineParams
from twinloop.io import load_image, load_mask
from twinloop.pipeline import run_pipeline

JOB_TIMEOUT_S = 15 * 60
FINAL_STATES = ("done", "failed")


class JobManager:
    def __init__(self, cases_root: Path):
        self.cases_root = Path(cases_root)
        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._pending = 0  # queued + running
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> None:
        self.cases_root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="twinloop-worker")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout)

    def recover(self) -> None:
        """Mark cases left unfinished by a previous server run as failed."""
        for status_file in self.cases_root.glob("*/status.json"):
            case_dir = status_file.parent
            try:
                if read_status(case_dir)["status"] not in FINAL_STATES:
                    update_status(case_dir, status="failed", error="Interrupted by server restart")
            except (OSError, ValueError, KeyError):
                continue

    # -- queue -------------------------------------------------------------------------------
    def submit(self, case_dir: Path) -> None:
        with self._lock:
            self._pending += 1
        self._queue.put(Path(case_dir))

    def queue_length(self) -> int:
        with self._lock:
            return self._pending

    # -- worker ------------------------------------------------------------------------------
    def _worker(self) -> None:
        while (case_dir := self._queue.get()) is not None:
            try:
                self._run(case_dir)
            finally:
                with self._lock:
                    self._pending -= 1

    def _run(self, case_dir: Path) -> None:
        t_job = time.monotonic()

        def check_time() -> None:
            if time.monotonic() - t_job > JOB_TIMEOUT_S:
                raise TimeoutError("Timed out after 15 minutes")

        def on_step(name: str) -> None:
            check_time()
            update_status(case_dir, status=name)

        try:
            status = update_status(case_dir, status="loading")
            image_path = next(p for p in case_dir.glob("input.*"))
            mask_path = next(case_dir.glob("input_mask.*"), None)
            t = time.perf_counter()
            image = load_image(image_path)
            mask = load_mask(mask_path, image) if mask_path else None
            load_s = time.perf_counter() - t
            solver = dataclasses.replace(
                PipelineParams().solver, aortic_pressure_mmhg=status["aortic_pressure_mmhg"]
            )
            result = run_pipeline(
                image,
                mask,
                PipelineParams(solver=solver),
                case_dir,
                status["case_id"],
                status["filename"],
                load_s=load_s,
                on_step=on_step,
            )
            check_time()
            update_status(case_dir, status="done", timings_s=result.timings_s)
        except Exception as exc:  # every failure ends in a visible `failed` status
            with open(case_dir / "log.txt", "a") as log:
                log.write(traceback.format_exc())
            update_status(case_dir, status="failed", error=str(exc))
