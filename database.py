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
            with open(DB_PATH, "r") as f:
                return json.load(f)
        return {"persons": {}, "guest_counter": 0}

    def _save(self):
        with open(DB_PATH, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_all_known(self) -> tuple[list[str], list[np.ndarray]]:
        """Return parallel (person_ids, encodings) lists for every stored face."""
        ids = []
        encodings = []
        for person_id, info in self._data["persons"].items():
            for enc in info["encodings"]:
                ids.append(person_id)
                encodings.append(np.array(enc, dtype=np.float64))
        return ids, encodings

    def add_person(
        self,
        encoding: np.ndarray,
        name: str | None = None,
        person_type: str = "guest",
        face_image: np.ndarray | None = None,
    ) -> str:
        """Add a new person. Returns the assigned person_id."""
        if person_type == "guest":
            self._data["guest_counter"] += 1
            person_id = f"guest_{self._data['guest_counter']:03d}"
            display_name = person_id
        else:
            person_id = name.lower().replace(" ", "_")
            display_name = name

        if person_id not in self._data["persons"]:
            self._data["persons"][person_id] = {
                "name": display_name,
                "type": person_type,
                "registered_at": datetime.now().isoformat(),
                "encodings": [],
            }

        self._data["persons"][person_id]["encodings"].append(encoding.tolist())

        if face_image is not None:
            self._save_face_image(person_id, face_image)

        self._save()
        return person_id

    def promote_to_worker(self, guest_id: str, name: str) -> str:
        """Re-key a guest entry as a named permanent worker."""
        if guest_id not in self._data["persons"]:
            raise ValueError(f"Person '{guest_id}' not found in database")

        new_id = name.lower().replace(" ", "_")
        person_data = self._data["persons"].pop(guest_id)
        person_data["name"] = name
        person_data["type"] = "worker"
        self._data["persons"][new_id] = person_data

        old_img = FACES_DIR / f"{guest_id}.jpg"
        if old_img.exists():
            old_img.rename(FACES_DIR / f"{new_id}.jpg")

        self._save()
        return new_id

    def get_person(self, person_id: str) -> dict | None:
        return self._data["persons"].get(person_id)

    def person_count(self) -> int:
        return len(self._data["persons"])

    def _save_face_image(self, person_id: str, image: np.ndarray):
        import cv2
        cv2.imwrite(str(FACES_DIR / f"{person_id}.jpg"), image)
