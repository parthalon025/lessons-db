"""Transfer-test evaluation pipeline (package)."""

# Transitional: re-export everything from backup while we extract modules
from lessons_db.eval._eval_backup import *  # noqa: F401,F403

# Wildcard import skips underscore-prefixed names; import them explicitly
from lessons_db.eval._eval_backup import (  # noqa: F401,F811
    _PRIOR_LOG_ODDS as _PRIOR_LOG_ODDS,
)
from lessons_db.eval._eval_backup import (
    _clean_principle as _clean_principle,
)
from lessons_db.eval._eval_backup import (
    _generate_for_lesson as _generate_for_lesson,
)
from lessons_db.eval.prompts import (  # noqa: F811
    _build_self_critique_prompt as _build_self_critique_prompt,
)
from lessons_db.eval.prompts import (
    build_binary_judge_prompt as build_binary_judge_prompt,
)
from lessons_db.eval.prompts import (
    build_generation_prompt as build_generation_prompt,
)
from lessons_db.eval.prompts import (
    build_judge_prompt as build_judge_prompt,
)
from lessons_db.eval.prompts import (
    build_mechanism_extraction_prompt as build_mechanism_extraction_prompt,
)
from lessons_db.eval.prompts import (
    build_paired_judge_prompt as build_paired_judge_prompt,
)
from lessons_db.eval.prompts import (
    build_simulation_prompt as build_simulation_prompt,
)
from lessons_db.eval.sampling import (  # noqa: F811
    _select_diverse as _select_diverse,
)
from lessons_db.eval.sampling import (
    select_source_lessons as select_source_lessons,
)
from lessons_db.eval.sampling import (
    select_transfer_targets as select_transfer_targets,
)

# Extracted modules (override backup imports):
from lessons_db.eval.variants import (  # noqa: F811
    DEFAULT_BINARY_JUDGE_MODEL as DEFAULT_BINARY_JUDGE_MODEL,
)
from lessons_db.eval.variants import (
    DEFAULT_JUDGE_MODEL as DEFAULT_JUDGE_MODEL,
)
from lessons_db.eval.variants import (
    VALID_GROUP_BY as VALID_GROUP_BY,
)
from lessons_db.eval.variants import (
    VARIANT_CONFIGS as VARIANT_CONFIGS,
)
