import argparse
import sys
from datetime import datetime, timedelta

from attendance import AttendanceTracker
from database import FaceDatabase

COOLDOWN_SEC = 10  # seconds before the same worker can toggle again


# ------------------------------------------------------------------
# Camera / attendance mode
# ------------------------------------------------------------------

def cmd_run(_args):
    import cv2
    import numpy as np
    from face_recognizer import FaceRecognizer

    db = FaceDatabase()
    attendance = AttendanceTracker()
    recognizer = FaceRecognizer(db)
    cooldowns: dict[str, datetime] = {}

    print("=== CV Attendance System ===")
    print(f"Workers on record: {db.worker_count()}")
    print("Camera starting... press 'q' to quit.\n")

    def on_known_face(person_id: str, person_name: str):
        now = datetime.now()
        if now < cooldowns.get(person_id, datetime.min):
            return
        new_state = attendance.toggle(person_id, person_name)
        cooldowns[person_id] = now + timedelta(seconds=COOLDOWN_SEC)
        greeting = "Welcome" if new_state == "in" else "Goodbye"
        print(f"[{now.strftime('%H:%M:%S')}] {greeting}, {person_name}! ({person_id})")

    def on_unknown_face(encoding, face_crop):
        if face_crop is not None and face_crop.size > 0:
            cv2.imshow("Unknown Face Detected", face_crop)
            cv2.waitKey(1500)
            cv2.destroyWindow("Unknown Face Detected")

        guest_id = db.add_guest(encoding, face_image=face_crop)
        print(f"\n[!] Unknown person detected — saved as '{guest_id}'")

        answer = _prompt("    Register as a permanent worker? (y/n): ")
        if answer == "y":
            name = _prompt("    Full name: ").strip()
            if name:
                worker_id = db.promote_guest_to_worker(guest_id, name)
                print(f"    Registered: '{name}'  (ID: {worker_id})\n")
            else:
                print(f"    No name provided — keeping as '{guest_id}'.\n")
        else:
            print(f"    Keeping as '{guest_id}'.\n")

    recognizer.run(
        on_unknown_face=on_unknown_face,
        on_known_face=on_known_face,
    )


# ------------------------------------------------------------------
# CLI subcommands
# ------------------------------------------------------------------

def cmd_list(_args):
    db = FaceDatabase()
    workers = db.list_workers()
    if not workers:
        print("No workers registered.")
        return
    print(f"\n{'ID':<6}  {'Name':<30}  Registered")
    print("-" * 56)
    for w in workers:
        print(f"{w['id']:<6}  {w['name']:<30}  {w['registered_at'][:10]}")
    print(f"\nTotal: {len(workers)} worker(s)")


def cmd_remove(args):
    db = FaceDatabase()
    identifier = " ".join(args.identifier)
    ok, name = db.remove_worker(identifier)
    if ok:
        print(f"Removed worker: {name}")
    else:
        print(f"Worker not found: '{identifier}'")
        print("Use 'python main.py list' to see valid IDs and names.")
        sys.exit(1)


def cmd_export(args):
    attendance = AttendanceTracker()
    target = args.date or None
    try:
        path = attendance.export_csv(target)
        print(f"CSV exported: {path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def _prompt(message: str) -> str:
    try:
        return input(message).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ""


def main():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="CV Attendance System",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Start the attendance camera (default)")

    sub.add_parser("list", help="List all registered workers")

    rm = sub.add_parser("remove", help="Remove a worker by ID or name")
    rm.add_argument(
        "identifier",
        nargs="+",
        help="Worker ID (e.g. W001) or full name (e.g. John Doe)",
    )

    exp = sub.add_parser("export", help="Export attendance CSV for a given date")
    exp.add_argument(
        "date",
        nargs="?",
        help="Date in YYYY-MM-DD format (default: today)",
    )

    args = parser.parse_args()

    dispatch = {
        None: cmd_run,
        "run": cmd_run,
        "list": cmd_list,
        "remove": cmd_remove,
        "export": cmd_export,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
