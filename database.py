import json
import numpy as np
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/db.json")
FACES_DIR = Path("data/faces")


class FaceDatabase:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if DB_PATH.exists():
            with open(DB_PATH) as f:
                raw = json.load(f)
            if "persons" in raw:
                return self._migrate(raw)
            return raw
        return {"next_id": 1, "free_ids": [], "workers": {}, "guests": {}, "guest_counter": 0}

    def _migrate(self, old: dict) -> dict:
        """One-time migration from the old name-keyed schema to numeric worker IDs."""
        new: dict = {"next_id": 1, "free_ids": [], "workers": {}, "guests": {}, "guest_counter": 0}
        for old_id, info in old.get("persons", {}).items():
            if info.get("type") == "worker":
                wid = new["next_id"]
                new["next_id"] += 1
                new["workers"][str(wid)] = {
                    "name": info["name"],
                    "registered_at": info.get("registered_at", datetime.now().isoformat()),
                    "encodings": info["encodings"],
                }
                old_img = FACES_DIR / f"{old_id}.jpg"
                if old_img.exists():
                    old_img.rename(FACES_DIR / f"{self.format_id(wid)}.jpg")
            else:
                new["guest_counter"] += 1
                gid = f"guest_{new['guest_counter']:03d}"
                new["guests"][gid] = {
                    "registered_at": info.get("registered_at", datetime.now().isoformat()),
                    "encodings": info["encodings"],
                }
        with open(DB_PATH, "w") as f:
            json.dump(new, f, indent=2)
        print("[db] Migrated database to numeric worker IDs.")
        return new

    def _save(self):
        with open(DB_PATH, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # ID allocation
    # ------------------------------------------------------------------

    @staticmethod
    def format_id(n: int) -> str:
        return f"W{n:03d}"

    def _parse_id(self, s: str) -> int | None:
        if s.upper().startswith("W"):
            try:
                return int(s[1:])
            except ValueError:
                pass
        return None

    def _next_available_id(self) -> int:
        if self._data["free_ids"]:
            return min(self._data["free_ids"])
        return self._data["next_id"]

    def _allocate_id(self, wid: int):
        if wid in self._data["free_ids"]:
            self._data["free_ids"].remove(wid)
        else:
            self._data["next_id"] = wid + 1

    # ------------------------------------------------------------------
    # Recognition feed
    # ------------------------------------------------------------------

    def get_all_known(self) -> tuple[list[str], list[np.ndarray]]:
        ids, encodings = [], []
        for key, info in self._data["workers"].items():
            for enc in info["encodings"]:
                ids.append(self.format_id(int(key)))
                encodings.append(np.array(enc, dtype=np.float64))
        for gid, info in self._data["guests"].items():
            for enc in info["encodings"]:
                ids.append(gid)
                encodings.append(np.array(enc, dtype=np.float64))
        return ids, encodings

    # ------------------------------------------------------------------
    # Guest management
    # ------------------------------------------------------------------

    def add_guest(self, encoding: np.ndarray, face_image: np.ndarray | None = None) -> str:
        self._data["guest_counter"] += 1
        gid = f"guest_{self._data['guest_counter']:03d}"
        self._data["guests"][gid] = {
            "registered_at": datetime.now().isoformat(),
            "encodings": [encoding.tolist()],
        }
        if face_image is not None:
            self._save_face_image(gid, face_image)
        self._save()
        return gid

    def promote_guest_to_worker(self, guest_id: str, name: str) -> str:
        if guest_id not in self._data["guests"]:
            raise ValueError(f"Guest '{guest_id}' not found")
        wid = self._next_available_id()
        self._allocate_id(wid)
        guest = self._data["guests"].pop(guest_id)
        self._data["workers"][str(wid)] = {
            "name": name,
            "registered_at": guest["registered_at"],
            "encodings": guest["encodings"],
        }
        old_img = FACES_DIR / f"{guest_id}.jpg"
        if old_img.exists():
            old_img.rename(FACES_DIR / f"{self.format_id(wid)}.jpg")
        self._save()
        return self.format_id(wid)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def add_worker(self, name: str, encoding: np.ndarray, face_image: np.ndarray | None = None) -> str:
        wid = self._next_available_id()
        self._allocate_id(wid)
        self._data["workers"][str(wid)] = {
            "name": name,
            "registered_at": datetime.now().isoformat(),
            "encodings": [encoding.tolist()],
        }
        if face_image is not None:
            self._save_face_image(self.format_id(wid), face_image)
        self._save()
        return self.format_id(wid)

    def remove_worker(self, identifier: str) -> tuple[bool, str]:
        """Remove by formatted ID (e.g. W001) or full name. Returns (success, removed_name)."""
        num = self._parse_id(identifier)
        if num is not None:
            key = str(num)
            if key in self._data["workers"]:
                name = self._data["workers"][key]["name"]
                del self._data["workers"][key]
                self._data["free_ids"].append(num)
                self._data["free_ids"].sort()
                (FACES_DIR / f"{self.format_id(num)}.jpg").unlink(missing_ok=True)
                self._save()
                return True, name

        target = identifier.strip().lower()
        for key, info in list(self._data["workers"].items()):
            if info["name"].lower() == target:
                num = int(key)
                name = info["name"]
                del self._data["workers"][key]
                self._data["free_ids"].append(num)
                self._data["free_ids"].sort()
                (FACES_DIR / f"{self.format_id(num)}.jpg").unlink(missing_ok=True)
                self._save()
                return True, name

        return False, ""

    def get_person(self, person_id: str) -> dict | None:
        """Unified lookup used by face_recognizer for display labels."""
        num = self._parse_id(person_id)
        if num is not None:
            info = self._data["workers"].get(str(num))
            if info:
                return {"name": info["name"], "type": "worker", "id": person_id}
        if person_id in self._data["guests"]:
            return {"name": person_id, "type": "guest", "id": person_id}
        return None

    def list_workers(self) -> list[dict]:
        return [
            {
                "id": self.format_id(int(k)),
                "name": v["name"],
                "registered_at": v["registered_at"],
            }
            for k, v in sorted(self._data["workers"].items(), key=lambda x: int(x[0]))
        ]

    def worker_count(self) -> int:
        return len(self._data["workers"])

    def person_count(self) -> int:
        return len(self._data["workers"]) + len(self._data["guests"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_face_image(self, person_id: str, image: np.ndarray):
        import cv2
        cv2.imwrite(str(FACES_DIR / f"{person_id}.jpg"), image)
