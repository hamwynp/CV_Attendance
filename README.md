# CV Attendance

A face recognition-based attendance system for workplaces. Workers show their face to a camera to clock in and out — no cards, no PINs. The system tracks daily presence and exports a CSV report with first-in time, last-out time, and total hours worked.

---

## How it works

The camera runs continuously at the entrance. When a worker's face is detected:

- If they are currently **out** → logs an **IN** event
- If they are currently **in** → logs an **OUT** event

This toggle fires once per appearance (10-second cooldown) so a single walk-past doesn't generate multiple events. Workers can enter and leave as many times as they need throughout the day; the system tracks every pair.

**New faces** are flagged on-screen and the operator is prompted to register them as a permanent worker. Workers are assigned a permanent numeric ID (`W001`, `W002`, ...). When a worker is removed, their ID is released back into the pool and reassigned to the next new hire.

---

## Features

- Real-time face detection and recognition via webcam
- Toggle-based clock-in / clock-out — no interaction needed beyond showing your face
- Permanent numeric worker IDs with reuse on removal
- Daily attendance log stored as JSON and exported to CSV
- Tracks multiple in/out cycles per day
- CLI for worker management — no GUI needed

---

## Requirements

- Python 3.10+
- A C++ compiler and `cmake` (required by `dlib`, which powers `face_recognition`)

**On Arch Linux:**
```bash
sudo pacman -S cmake base-devel
```

**On Ubuntu / Debian:**
```bash
sudo apt install cmake build-essential
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Start the attendance camera

```bash
python main.py
# or explicitly:
python main.py run
```

The camera opens and begins recognizing faces. When an unknown face appears, the terminal prompts the operator to register them. Press `q` to quit.

### Manage workers

```bash
# List all registered workers
python main.py list

# Remove a worker by ID
python main.py remove W001

# Remove a worker by name
python main.py remove "Jane Doe"
```

### Export attendance

```bash
# Export today's CSV
python main.py export

# Export a specific date
python main.py export 2026-06-27
```

---

## CSV output

Reports are written to `data/attendance/YYYY-MM-DD.csv` automatically at midnight and on demand via `export`. Each row represents one worker's day:

| Column | Description |
|---|---|
| `date` | Date of the record |
| `worker_id` | Permanent worker ID (e.g. `W001`) |
| `name` | Worker's full name |
| `first_in` | Time of first arrival |
| `last_out` | Time of last departure |
| `total_hours` | Sum of all completed in→out intervals |

Workers who are still clocked in when the export runs will have `last_out` and `total_hours` left blank for that day.

---

## Project structure

```
CV_Attendance/
├── main.py             # Entry point and CLI (run / list / remove / export)
├── face_recognizer.py  # Camera loop and face matching
├── database.py         # Worker registry with numeric IDs and free-ID pool
├── attendance.py       # Daily event log, toggle logic, and CSV export
├── requirements.txt
└── data/
    ├── db.json         # Worker database (gitignored — contains face encodings)
    ├── faces/          # Saved face images (gitignored)
    └── attendance/     # Daily JSON logs and CSV exports (gitignored)
```

---

## Worker ID rules

- IDs are assigned sequentially starting at `W001`
- An ID is **permanent** — it never changes for a given worker
- When a worker is removed, their ID returns to the pool and is reissued to the next new registration (lowest available number first)
- IDs cannot be manually set or renamed

---

## Data privacy

All face encodings, worker photos, and attendance records are stored locally and excluded from version control via `.gitignore`. Nothing is sent to any external service.
