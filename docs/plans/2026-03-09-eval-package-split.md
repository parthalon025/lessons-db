# eval.py → eval/ Package Split

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 2100-line `eval.py` monolith into a 9-module package so sub-agents can read each file in one pass (all under 500 lines).

**Architecture:** Convert `src/lessons_db/eval.py` to `src/lessons_db/eval/__init__.py` that re-exports all public symbols from 9 internal modules. Each module corresponds to an existing section separator in the file. Zero behavior change — every existing `from lessons_db.eval import X` continues to work.

**Tech Stack:** Python 3.14, pytest, lessons-db

---

### Task 1: Create eval/ package scaffold

**Files:**
- Delete: `src/lessons_db/eval.py`
- Create: `src/lessons_db/eval/__init__.py`
- Create: `src/lessons_db/eval/variants.py`

**Step 1: Rename eval.py to eval/ package**

```bash
cd ~/Documents/projects/lessons-db
mv src/lessons_db/eval.py src/lessons_db/_eval_backup.py
mkdir -p src/lessons_db/eval
```

**Step 2: Create variants.py**

Extract lines 1-104 from `_eval_backup.py` into `src/lessons_db/eval/variants.py`.

Contents: module docstring, all stdlib imports needed by this file only (`logging`, `typing`), then:
- `_log` (module-level logger — use `__name__`)
- `DEFAULT_JUDGE_MODEL`
- `DEFAULT_BINARY_JUDGE_MODEL`
- `_RETRYABLE_CODES`
- `_MAX_RETRIES`
- `_RETRY_BASE_DELAY`
- `VALID_GROUP_BY`
- `VARIANT_CONFIGS` (the full dict, lines 35-104)

This module has **zero internal imports** — it only uses stdlib.

**Step 3: Create `__init__.py` with re-exports**

Create `src/lessons_db/eval/__init__.py` that imports and re-exports every public symbol from variants.py:

```python
"""Transfer-test evaluation pipeline (package)."""

from lessons_db.eval.variants import (
    DEFAULT_BINARY_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL,
    VALID_GROUP_BY,
    VARIANT_CONFIGS,
)

__all__ = [
    "DEFAULT_BINARY_JUDGE_MODEL",
    "DEFAULT_JUDGE_MODEL",
    "VALID_GROUP_BY",
    "VARIANT_CONFIGS",
]
```

(We will extend `__init__.py` in every subsequent task.)

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

Expected: PASS (tests import from `lessons_db.eval` which `__init__.py` serves).

If any test fails due to missing imports, the symbol hasn't been re-exported yet — that's expected; we'll add the rest in subsequent tasks.

Actually — at this point most symbols are missing. To keep tests passing after every task, keep `_eval_backup.py` in place and have `__init__.py` do a wildcard re-export from it as a transitional bridge:

```python
"""Transfer-test evaluation pipeline (package)."""

# Transitional: re-export everything from backup while we extract modules
from lessons_db.eval._eval_backup import *  # noqa: F401,F403

# Extracted modules (override backup imports):
from lessons_db.eval.variants import (  # noqa: F811
    DEFAULT_BINARY_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL,
    VALID_GROUP_BY,
    VARIANT_CONFIGS,
)
```

**Step 5: Run full test suite**

```bash
pytest --timeout=120 -x -q -n 6
```

Expected: ALL PASS (967 tests). The wildcard import from `_eval_backup` keeps everything working.

**Step 6: Commit**

```bash
git add src/lessons_db/eval/ src/lessons_db/_eval_backup.py
git add -u  # catches the deleted eval.py
git commit -m "refactor(eval): create eval/ package scaffold with transitional bridge"
```

---

### Task 2: Extract sampling.py

**Files:**
- Create: `src/lessons_db/eval/sampling.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py` (delete extracted functions)

**Step 1: Create sampling.py**

Extract these functions from `_eval_backup.py` (lines 112-264):
- `select_source_lessons` (112-165)
- `_select_diverse` (168-193)
- `select_transfer_targets` (196-264)

Imports needed:
```python
import logging
import sqlite3
from typing import Any

from lessons_db.eval.variants import VALID_GROUP_BY

_log = logging.getLogger(__name__)
```

**Step 2: Delete these functions from `_eval_backup.py`**

Remove the function bodies and their section header comments (lines 107-265). Keep section separators for context if helpful, or just delete the whole block.

**Step 3: Add re-exports to `__init__.py`**

Add after the variants import block:
```python
from lessons_db.eval.sampling import (
    select_source_lessons,
    select_transfer_targets,
    _select_diverse,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

Expected: ALL PASS.

**Step 5: Commit**

```bash
git add src/lessons_db/eval/sampling.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract sampling.py"
```

---

### Task 3: Extract prompts.py

**Files:**
- Create: `src/lessons_db/eval/prompts.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py`

**Step 1: Create prompts.py**

Extract these functions (lines 267-436):
- `build_generation_prompt` (272-297)
- `_build_fewshot_prompt` (300-325)
- `_build_zero_shot_prompt` (328-348)
- `_build_chunked_prompt` (351-371)
- `_build_contrastive_prompt` (374-408)
- `_build_self_critique_prompt` (411-436)

Also extract judge/binary/paired/mechanism/simulation prompt builders from their later sections:
- `build_judge_prompt` (866-906)
- `build_binary_judge_prompt` (1096-1126)
- `build_paired_judge_prompt` (1149-1191)
- `build_mechanism_extraction_prompt` (1366-1391)
- `build_simulation_prompt` (1877-1894)

Imports needed:
```python
import random
from typing import Any

from lessons_db.eval.variants import VARIANT_CONFIGS
```

No `_log` needed — prompt builders don't log.

**Step 2: Delete these functions from `_eval_backup.py`**

**Step 3: Add re-exports to `__init__.py`**

```python
from lessons_db.eval.prompts import (
    build_binary_judge_prompt,
    build_generation_prompt,
    build_judge_prompt,
    build_mechanism_extraction_prompt,
    build_paired_judge_prompt,
    build_simulation_prompt,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

**Step 5: Commit**

```bash
git add src/lessons_db/eval/prompts.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract prompts.py"
```

---

### Task 4: Extract client.py

**Files:**
- Create: `src/lessons_db/eval/client.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py`

**Step 1: Create client.py**

Extract (lines 439-570 + 940-985):
- `_parse_ollama_response` (444-452)
- `call_ollama` (455-520)
- `_clean_principle` (523-570)
- `call_judge` (940-956)
- `_call_openai` (959-985)

Imports needed:
```python
import json as _json
import logging
import re as _re
import time
import urllib.error
import urllib.request
from typing import Any

from lessons_db.eval.variants import (
    _MAX_RETRIES,
    _RETRY_BASE_DELAY,
    _RETRYABLE_CODES,
)

_log = logging.getLogger(__name__)
```

**Step 2: Delete from `_eval_backup.py`**

**Step 3: Add re-exports to `__init__.py`**

```python
from lessons_db.eval.client import (
    _clean_principle,
    _parse_ollama_response,
    call_judge,
    call_ollama,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

**Step 5: Commit**

```bash
git add src/lessons_db/eval/client.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract client.py"
```

---

### Task 5: Extract generate.py

**Files:**
- Create: `src/lessons_db/eval/generate.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py`

**Step 1: Create generate.py**

Extract (lines 573-858):
- `_load_resume_state` (578-586)
- `_load_siblings_by_cluster` (589-610)
- `_generate_mechanism` (613-680)
- `_generate_for_lesson` (683-760)
- `_save_results` (763-783)
- `run_eval_generate` (786-858)

Imports needed:
```python
import json as _json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lessons_db.eval.client import _clean_principle, call_ollama
from lessons_db.eval.prompts import (
    _build_self_critique_prompt,
    build_generation_prompt,
    build_mechanism_extraction_prompt,
)
from lessons_db.eval.sampling import select_source_lessons
from lessons_db.eval.variants import VARIANT_CONFIGS

_log = logging.getLogger(__name__)
```

Note: `_generate_mechanism` calls `build_mechanism_extraction_prompt` and `call_ollama`. `_generate_for_lesson` calls `build_generation_prompt`, `call_ollama`, `_build_self_critique_prompt`, and `_clean_principle`. `run_eval_generate` calls `select_source_lessons`, `_load_siblings_by_cluster`, `_generate_for_lesson`, `_generate_mechanism`, `_load_resume_state`, `_save_results`.

**Step 2: Delete from `_eval_backup.py`**

**Step 3: Add re-exports to `__init__.py`**

```python
from lessons_db.eval.generate import (
    _generate_for_lesson,
    run_eval_generate,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

**Step 5: Commit**

```bash
git add src/lessons_db/eval/generate.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract generate.py"
```

---

### Task 6: Extract signals.py

**Files:**
- Create: `src/lessons_db/eval/signals.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py`

**Step 1: Create signals.py**

Extract (lines 1394-1560):
- `parse_mechanism_triplet` (1394-1410)
- `compute_paired_signal` (1418-1426)
- `compute_embedding_signal` (1429-1445)
- `compute_scope_signal` (1448-1461)
- `compute_mechanism_signal` (1464-1475)
- `_PRIOR_LOG_ODDS` (1483)
- `compute_transfer_posterior` (1486-1499)
- `compute_bayesian_metrics` (1507-1560)

Imports needed:
```python
import math
import re as _re
from typing import Any
```

No internal imports — this module is self-contained (pure math).

**Step 2: Delete from `_eval_backup.py`**

**Step 3: Add re-exports to `__init__.py`**

```python
from lessons_db.eval.signals import (
    compute_bayesian_metrics,
    compute_embedding_signal,
    compute_mechanism_signal,
    compute_paired_signal,
    compute_scope_signal,
    compute_transfer_posterior,
    parse_mechanism_triplet,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

**Step 5: Commit**

```bash
git add src/lessons_db/eval/signals.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract signals.py"
```

---

### Task 7: Extract judge.py

**Files:**
- Create: `src/lessons_db/eval/judge.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py`

**Step 1: Create judge.py**

Extract parsers, metrics, and tournament runner:
- `parse_judge_scores` (909-932)
- `parse_binary_judge` (1129-1146)
- `parse_paired_judge` (1194-1211)
- `compute_metrics` (993-1047)
- `compute_rank_metrics` (1050-1093)
- `compute_tournament_metrics` (1320-1358)
- `run_paired_tournament` (1214-1317)
- `run_eval_judge` (1697-1797)

Imports needed:
```python
import json as _json
import logging
import re as _re
import sqlite3
from pathlib import Path
from typing import Any

from lessons_db.eval.client import call_judge
from lessons_db.eval.prompts import (
    build_binary_judge_prompt,
    build_judge_prompt,
    build_paired_judge_prompt,
)
from lessons_db.eval.sampling import select_transfer_targets
from lessons_db.eval.variants import VARIANT_CONFIGS

_log = logging.getLogger(__name__)
```

Note: `run_eval_judge` calls `select_transfer_targets`, `build_judge_prompt`, `build_binary_judge_prompt`, `call_judge`, `parse_judge_scores`, `parse_binary_judge`, `compute_metrics`. `run_paired_tournament` calls `select_transfer_targets`, `build_paired_judge_prompt`, `call_judge`, `parse_paired_judge`.

**Step 2: Delete from `_eval_backup.py`**

**Step 3: Add re-exports to `__init__.py`**

```python
from lessons_db.eval.judge import (
    compute_metrics,
    compute_rank_metrics,
    compute_tournament_metrics,
    parse_binary_judge,
    parse_judge_scores,
    parse_paired_judge,
    run_eval_judge,
    run_paired_tournament,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

**Step 5: Commit**

```bash
git add src/lessons_db/eval/judge.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract judge.py"
```

---

### Task 8: Extract reports.py

**Files:**
- Create: `src/lessons_db/eval/reports.py`
- Modify: `src/lessons_db/eval/__init__.py`
- Modify: `src/lessons_db/eval/_eval_backup.py`

**Step 1: Create reports.py**

Extract all report rendering + reference comparison + simulation:
- `_render_failure_binary` (1563-1582)
- `_render_failure_rubric` (1585-1596)
- `_render_pair_sections` (1599-1628)
- `render_report` (1631-1689)
- `diagnose_vs_reference` (1805-1869)
- `parse_simulation_result` (1897-1902)
- `compute_simulation_lift` (1910-1938)
- `_render_v2_failure_analysis` (1946-1973)
- `_render_v2_tournament` (1976-1988)
- `_render_v2_bayesian` (1991-2012)
- `_render_v2_signal_diagnostics` (2015-2046)
- `render_v2_report` (2049-2100)

Imports needed:
```python
from datetime import UTC, datetime
from typing import Any

from lessons_db.eval.variants import VARIANT_CONFIGS
```

**Step 2: Delete from `_eval_backup.py`**

At this point `_eval_backup.py` should be empty (only imports and maybe empty section comments).

**Step 3: Add re-exports to `__init__.py`**

```python
from lessons_db.eval.reports import (
    compute_simulation_lift,
    diagnose_vs_reference,
    parse_simulation_result,
    render_report,
    render_v2_report,
)
```

**Step 4: Run tests**

```bash
pytest tests/test_eval.py -x -q --timeout=120
```

**Step 5: Commit**

```bash
git add src/lessons_db/eval/reports.py src/lessons_db/eval/__init__.py src/lessons_db/eval/_eval_backup.py
git commit -m "refactor(eval): extract reports.py"
```

---

### Task 9: Clean up — remove backup, finalize __init__.py

**Files:**
- Delete: `src/lessons_db/eval/_eval_backup.py`
- Modify: `src/lessons_db/eval/__init__.py`

**Step 1: Delete the backup file**

```bash
rm src/lessons_db/eval/_eval_backup.py
```

**Step 2: Clean up `__init__.py`**

Remove the transitional wildcard import line:
```python
from lessons_db.eval._eval_backup import *  # noqa: F401,F403
```

The file should now contain only explicit re-exports from the 8 modules. Add a complete `__all__` list with every public symbol.

**Step 3: Run full test suite**

```bash
pytest --timeout=120 -x -q -n 6
```

Expected: ALL 967 tests pass.

**Step 4: Run linter**

```bash
make lint
```

Fix any import ordering or unused import warnings.

**Step 5: Commit**

```bash
git add -u
git commit -m "refactor(eval): remove transitional backup, finalize eval/ package"
```

---

### Task 10: Full suite verification + final commit

**Step 1: Run full test suite with parallel**

```bash
pytest --timeout=120 -x -q -n 6
```

Expected: ALL PASS.

**Step 2: Run lint**

```bash
make lint
```

Expected: Clean.

**Step 3: Verify file sizes**

```bash
wc -l src/lessons_db/eval/*.py
```

Expected: Every module under 500 lines, total ~2200 lines (slight overhead from module-level imports).

**Step 4: Verify imports work**

```bash
source .venv/bin/activate
python3 -c "from lessons_db.eval import VARIANT_CONFIGS, call_ollama, run_eval_generate, render_v2_report; print('OK')"
```

**Step 5: Squash or keep commits (user preference)**

All done. The eval package is split.
