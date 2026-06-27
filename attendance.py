import csv
import json
from datetime import date, datetime
from pathlib import Path
from threading import Lock

ATTENDANCE_DIR = Path("data/attendance")


class AttendanceTracker:
    def __init__(self):
        ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._today = date.today().isoformat()
        self._events: dict[str, list[dict]] = {}
        self._state: dict[str, str] = {}  # worker_id -> "in" | "out"
        self._load_today()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _today_path(self) -> Path:
        return ATTENDANCE_DIR / f"{self._today}.json"

    def _load_today(self):
        p = self._today_path()
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            self._events = data.get("events", {})
            for wid, events in self._events.items():
                if events:
                    self._state[wid] = events[-1]["type"]

    def _save(self):
        with open(self._today_path(), "w") as f:
            json.dump({"date": self._today, "events": self._events}, f, indent=2)

    def _check_rollover(self):
        today = date.today().isoformat()
        if today != self._today:
            self._finalize_day(self._today)
            self._today = today
            self._events = {}
            self._state = {}

    def _finalize_day(self, target_date: str):
        """Close any open IN sessions and auto-export CSV for the completed day."""
        changed = False
        for wid, state in list(self._state.items()):
            if state == "in":
                evts = self._events.get(wid, [])
                last_in = next((e["time"] for e in reversed(evts) if e["type"] == "in"), None)
                if last_in:
                    evts.append({"type": "out", "time": last_in})
                    changed = True
        if changed:
            self._save()
        try:
            self.export_csv(target_date)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core: toggle IN/OUT
    # ------------------------------------------------------------------

    def toggle(self, worker_id: str, worker_name: str) -> str:
        """
        Toggle the worker's presence state.
        Returns the NEW state: "in" or "out".
        """
        with self._lock:
            self._check_rollover()
            now = datetime.now()
            new_state = "in" if self._state.get(worker_id, "out") == "out" else "out"

            if worker_id not in self._events:
                self._events[worker_id] = []

            self._events[worker_id].append({
                "type": new_state,
                "time": now.strftime("%H:%M:%S"),
                "name": worker_name,
            })
            self._state[worker_id] = new_state
            self._save()
            return new_state

    def get_state(self, worker_id: str) -> str:
        return self._state.get(worker_id, "out")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self, target_date: str | None = None) -> Path:
        """
        Export attendance for target_date (YYYY-MM-DD) to CSV.
        Columns: date, worker_id, name, first_in, last_out, total_hours
        total_hours is the sum of all completed IN→OUT pairs.
        """
        if target_date is None:
            with self._lock:
                target_date = self._today

        json_path = ATTENDANCE_DIR / f"{target_date}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No attendance data for {target_date}")

        with open(json_path) as f:
            data = json.load(f)

        rows = []
        for worker_id, events in data["events"].items():
            name = next((e["name"] for e in events if e.get("name")), "")
            in_times = [e["time"] for e in events if e["type"] == "in"]
            out_times = [e["time"] for e in events if e["type"] == "out"]

            first_in = in_times[0] if in_times else ""
            last_out = out_times[-1] if out_times else ""

            total_seconds = 0.0
            fmt = "%H:%M:%S"
            for in_t, out_t in zip(in_times, out_times):
                delta = (
                    datetime.strptime(out_t, fmt) - datetime.strptime(in_t, fmt)
                ).total_seconds()
                total_seconds += max(0.0, delta)

            total_hours = round(total_seconds / 3600, 2) if total_seconds > 0 else ""

            rows.append({
                "date": target_date,
                "worker_id": worker_id,
                "name": name,
                "first_in": first_in,
                "last_out": last_out,
                "total_hours": total_hours,
            })

        rows.sort(key=lambda r: r["worker_id"])

        csv_path = ATTENDANCE_DIR / f"{target_date}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "worker_id", "name", "first_in", "last_out", "total_hours"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return csv_path
