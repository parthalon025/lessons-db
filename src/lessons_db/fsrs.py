"""FSRS spaced repetition scheduling for lessons-db.

Implements core FSRS equations for retrievability, stability, and difficulty
updates. No external dependencies — pure math from the FSRS algorithm with
optimized default parameters.

The implementation follows the reference at:
    https://borretti.me/article/implementing-fsrs-in-100-lines
    https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

Weight mapping (19 parameters, w0-w18):
    w0-w3:  initial stability per grade (Again, Hard, Good, Easy)
    w4:     initial difficulty anchor
    w5:     initial difficulty grade sensitivity
    w6:     difficulty update rate
    w7:     difficulty mean reversion strength
    w8:     stability success exponential factor
    w9:     stability-S power (success)
    w10:    stability-R sensitivity (success)
    w11:    lapse stability constant
    w12:    lapse D power
    w13:    lapse S power
    w14:    lapse R sensitivity
    w15:    Hard modifier (<1 dampens success boost)
    w16:    Easy modifier (>1 amplifies success boost)
    w17-w18: same-day review (not used in lessons-db)
"""

import logging
import math
import sqlite3
from datetime import date

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grade constants — map to lesson outcomes
# ---------------------------------------------------------------------------

GRADE_AGAIN = 1  # dismissed / wrong
GRADE_HARD = 2
GRADE_GOOD = 3  # heeded / correct
GRADE_EASY = 4  # false_positive / remove from rotation

_VALID_GRADES = {GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY}

# ---------------------------------------------------------------------------
# FSRS default parameters (optimized from large-scale user data)
# ---------------------------------------------------------------------------

W = [
    0.40255,
    1.18385,
    3.173,
    15.69105,  # w0-w3: initial S per grade
    7.1949,  # w4: initial D anchor
    0.5345,  # w5: initial D grade sensitivity
    1.4604,  # w6: difficulty update rate
    0.0046,  # w7: difficulty mean reversion
    1.54575,  # w8: success exponential factor
    0.1192,  # w9: success S power
    1.01925,  # w10: success R sensitivity
    1.9395,  # w11: lapse constant
    0.11,  # w12: lapse D power
    0.29605,  # w13: lapse S power
    2.2698,  # w14: lapse R sensitivity
    0.2315,  # w15: Hard modifier
    2.9898,  # w16: Easy modifier
    0.51655,  # w17: same-day factor (unused)
    0.6621,  # w18: same-day grade (unused)
]

# Initial stability by grade (first review) — directly from w0-w3
INITIAL_S = {
    GRADE_AGAIN: W[0],
    GRADE_HARD: W[1],
    GRADE_GOOD: W[2],
    GRADE_EASY: W[3],
}

# Retrievability curve constants (FSRS-5 power-law forgetting curve)
# R(t,S) = (1 + FACTOR * t/S) ^ DECAY
# Calibrated so R(S, S) = 0.9 (90% retention at t=S)
DECAY: float = -0.5
FACTOR: float = 19.0 / 81.0  # 0.9^(1/DECAY) - 1


# ---------------------------------------------------------------------------
# Core FSRS equations
# ---------------------------------------------------------------------------


def compute_retrievability(stability: float, days_since_review: float) -> float:
    """Compute memory retrievability R given stability S and elapsed time t.

    Formula: R = (1 + FACTOR * t/S)^DECAY

    Where FACTOR=19/81, DECAY=-0.5, ensuring R(S,S)=0.9.
    Returns 1.0 when t=0. Monotonically decreasing.
    """
    if days_since_review <= 0.0:
        return 1.0
    return (1.0 + FACTOR * days_since_review / stability) ** DECAY


def _clamp_d(d: float) -> float:
    """Clamp difficulty to valid range [1.0, 10.0]."""
    return max(1.0, min(10.0, d))


def _initial_difficulty(grade: int) -> float:
    """Compute initial difficulty D0 for first review.

    D0(G) = w4 - e^(w5*(G-1)) + 1, clamped to [1, 10].
    """
    return _clamp_d(W[4] - math.exp(W[5] * (grade - 1)) + 1.0)


def update_difficulty(old_D: float, grade: int) -> float:
    """Compute new difficulty after a review.

    Three steps:
    1. delta_d = -w6 * (grade - 3)      [grade-dependent shift]
    2. D' = D + delta_d * ((10 - D)/9)  [linear damping near 10]
    3. D'' = w7*D0(Easy) + (1-w7)*D'    [mean reversion to Easy baseline]

    Result clamped to [1.0, 10.0].
    """
    # Grade-dependent change
    delta = -W[6] * (grade - 3)  # Again -> +2*w6, Hard -> +w6, Good -> 0, Easy -> -w6
    # Apply with linear damping
    d_prime = old_D + delta * ((10.0 - old_D) / 9.0)
    # Mean reversion toward D0(Easy)
    d0_easy = _initial_difficulty(GRADE_EASY)
    new_d = W[7] * d0_easy + (1.0 - W[7]) * d_prime
    return _clamp_d(new_d)


def update_stability(old_S: float, old_D: float, grade: int, R: float) -> float:
    """Compute new stability after a review.

    For successful reviews (Hard/Good/Easy): stability increases.
    For Again (lapse): stability decreases.
    """
    if grade == GRADE_AGAIN:
        return _stability_after_lapse(old_S, old_D, R)
    return _stability_after_success(old_S, old_D, R, grade)


def _stability_after_success(old_S: float, old_D: float, R: float, grade: int) -> float:
    """FSRS stability update for successful reviews (grade >= 2).

    S' = S * (1 + e^(w8) * (11-D) * S^(-w9) * (e^(w10*(1-R))-1) * h * b)

    h = w15 if Hard else 1.0
    b = w16 if Easy else 1.0
    """
    h = W[15] if grade == GRADE_HARD else 1.0
    b = W[16] if grade == GRADE_EASY else 1.0

    alpha = 1.0 + math.exp(W[8]) * (11.0 - old_D) * old_S ** (-W[9]) * (math.exp(W[10] * (1.0 - R)) - 1.0) * h * b
    new_s = old_S * alpha
    return max(new_s, 0.01)  # floor to prevent degenerate values


def _stability_after_lapse(old_S: float, old_D: float, R: float) -> float:
    """FSRS stability update after a lapse (Again grade).

    S' = min(w11 * D^(-w12) * ((S+1)^w13 - 1) * e^(w14*(1-R)), S)

    Post-lapse stability cannot exceed pre-lapse stability.
    """
    new_s = W[11] * old_D ** (-W[12]) * ((old_S + 1.0) ** W[13] - 1.0) * math.exp(W[14] * (1.0 - R))
    # Lapse can never increase stability
    new_s = min(new_s, old_S)
    return max(new_s, 0.01)  # floor


# ---------------------------------------------------------------------------
# DB schema extension
# ---------------------------------------------------------------------------


def ensure_fsrs_columns(conn: sqlite3.Connection) -> None:
    """Add FSRS columns to lessons table (idempotent).

    Adds: stability REAL, difficulty REAL, last_review_date TEXT
    """
    columns = [
        ("stability", "REAL"),
        ("difficulty", "REAL"),
        ("last_review_date", "TEXT"),
    ]
    for col_name, col_type in columns:
        try:
            conn.execute(f"ALTER TABLE lessons ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise


# ---------------------------------------------------------------------------
# DB integration
# ---------------------------------------------------------------------------


def get_due_lessons(conn: sqlite3.Connection, threshold: float = 0.9) -> list[dict]:
    """Return lessons whose computed retrievability R is below threshold.

    Only includes lessons that have been reviewed at least once
    (last_review_date IS NOT NULL and stability IS NOT NULL).

    Returns list of dicts with lesson fields plus computed 'retrievability'.
    Results are sorted by retrievability ascending (most forgotten first).
    """
    rows = conn.execute(
        """
        SELECT id, title, one_liner, stability, difficulty, last_review_date,
               severity, cluster, enforcement
        FROM lessons
        WHERE last_review_date IS NOT NULL
          AND stability IS NOT NULL
          AND stability > 0
        """,
    ).fetchall()

    today = date.today()
    due = []
    for row in rows:
        review_date = date.fromisoformat(row["last_review_date"])
        days_elapsed = (today - review_date).days
        r = compute_retrievability(row["stability"], float(days_elapsed))
        if r < threshold:
            entry = dict(row)
            entry["retrievability"] = r
            entry["days_since_review"] = days_elapsed
            due.append(entry)

    # Most forgotten first
    due.sort(key=lambda x: x["retrievability"])
    return due


def record_review(conn: sqlite3.Connection, lesson_id: int, grade: int) -> dict:
    """Record a review of a lesson, updating its FSRS parameters.

    For the first review: uses INITIAL_S and _initial_difficulty().
    For subsequent reviews: applies FSRS update equations.

    Returns dict with updated stability, difficulty, retrievability, last_review_date.

    Raises ValueError for invalid lesson_id or grade.
    """
    if grade not in _VALID_GRADES:
        raise ValueError(f"Grade must be one of {sorted(_VALID_GRADES)}, got {grade}")

    row = conn.execute(
        "SELECT stability, difficulty, last_review_date FROM lessons WHERE id = ?",
        [lesson_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"Lesson {lesson_id} not found")

    old_s = row["stability"]
    old_d = row["difficulty"]
    old_review = row["last_review_date"]

    today = date.today().isoformat()

    if old_s is None or old_review is None:
        # First review — use initial values
        new_s = INITIAL_S[grade]
        new_d = _initial_difficulty(grade)
    else:
        # Subsequent review — compute R then update
        days_elapsed = (date.today() - date.fromisoformat(old_review)).days
        r = compute_retrievability(old_s, float(days_elapsed))
        new_s = update_stability(old_s, old_d, grade, r)
        new_d = update_difficulty(old_d, grade)

    conn.execute(
        "UPDATE lessons SET stability = ?, difficulty = ?, last_review_date = ? WHERE id = ?",
        [new_s, new_d, today, lesson_id],
    )
    conn.commit()

    _log.info(
        "record_review: lesson=%d grade=%d S=%.3f D=%.3f",
        lesson_id,
        grade,
        new_s,
        new_d,
    )

    return {
        "stability": new_s,
        "difficulty": new_d,
        "retrievability": 1.0,  # just reviewed
        "last_review_date": today,
    }
