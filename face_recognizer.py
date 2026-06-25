import cv2
import face_recognition
import numpy as np
from database import FaceDatabase


TOLERANCE = 0.55       # Lower = stricter; 0.55 balances accuracy vs false rejects
PROCESS_SCALE = 0.25   # Downscale factor for faster detection per frame


class FaceRecognizer:
    def __init__(self, db: FaceDatabase):
        self.db = db
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _open_camera(self, index: int = 0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")

    def _release_camera(self):
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._cap = None

    # ------------------------------------------------------------------
    # Core recognition
    # ------------------------------------------------------------------

    def recognize_frame(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect and identify all faces in a BGR frame.

        Returns a list of tuples:
            (top, right, bottom, left, encoding, person_id_or_None)
        where coordinates are in the original frame's pixel space.
        """
        small = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        if not locations:
            return []

        encodings = face_recognition.face_encodings(rgb_small, locations)
        known_ids, known_encodings = self.db.get_all_known()

        scale = int(1 / PROCESS_SCALE)
        results = []

        for (top, right, bottom, left), enc in zip(locations, encodings):
            person_id = None

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                best = int(np.argmin(distances))
                if distances[best] <= TOLERANCE:
                    person_id = known_ids[best]

            results.append((
                top * scale,
                right * scale,
                bottom * scale,
                left * scale,
                enc,
                person_id,
            ))

        return results

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, on_unknown_face):
        """
        Start the live recognition loop.

        on_unknown_face(encoding, face_crop_bgr) is called (with the camera
        released) whenever a face cannot be matched to any known person.
        """
        self._open_camera()
        print("Camera ready — press 'q' to quit.\n")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    print("[!] Failed to read from camera.")
                    break

                results = self.recognize_frame(frame)

                unknown_found = False
                for top, right, bottom, left, encoding, person_id in results:
                    if person_id is None:
                        unknown_found = True
                        face_crop = frame[top:bottom, left:right].copy()
                        self._release_camera()
                        on_unknown_face(encoding, face_crop)
                        self._open_camera()
                        break  # re-evaluate after registration

                    self._draw_label(frame, top, right, bottom, left, person_id)

                if not unknown_found:
                    cv2.imshow("CV Attendance", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._release_camera()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_label(self, frame, top, right, bottom, left, person_id):
        person = self.db.get_person(person_id)
        label = person["name"] if person else person_id
        color = (0, 200, 0) if person and person["type"] == "worker" else (0, 140, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 32), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            frame, label, (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1,
        )
