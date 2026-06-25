import cv2
import numpy as np
from database import FaceDatabase
from face_recognizer import FaceRecognizer


def handle_unknown_face(db: FaceDatabase, encoding: np.ndarray, face_crop: np.ndarray):
    """
    Called (with the camera released) when an unrecognized face is detected.
    Saves them as guest_xxx, then optionally promotes to permanent worker.
    """
    if face_crop is not None and face_crop.size > 0:
        cv2.imshow("Unknown Face Detected", face_crop)
        cv2.waitKey(1500)
        cv2.destroyWindow("Unknown Face Detected")

    guest_id = db.add_person(encoding, person_type="guest", face_image=face_crop)
    print(f"\n[!] Unknown person detected — saved as '{guest_id}'")

    answer = _prompt("    Register as a permanent worker? (y/n): ")
    if answer == "y":
        name = _prompt("    Full name: ").strip()
        if name:
            worker_id = db.promote_to_worker(guest_id, name)
            print(f"    Registered worker: '{name}'  (id: {worker_id})\n")
        else:
            print(f"    No name provided — keeping as '{guest_id}'.\n")
    else:
        print(f"    Keeping as '{guest_id}'.\n")


def _prompt(message: str) -> str:
    try:
        return input(message).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ""


def main():
    db = FaceDatabase()
    recognizer = FaceRecognizer(db)

    print("=== CV Attendance System ===")
    print(f"Database: {db.person_count()} person(s) on record.")
    print("Starting camera...\n")

    recognizer.run(
        on_unknown_face=lambda enc, crop: handle_unknown_face(db, enc, crop)
    )


if __name__ == "__main__":
    main()
