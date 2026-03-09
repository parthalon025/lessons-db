"""Transfer-test evaluation pipeline (package).

Re-exports all public symbols so ``from lessons_db.eval import X`` keeps working.
"""

# --- analysis ---
from lessons_db.eval.analysis import (
    compute_per_lesson_breakdown as compute_per_lesson_breakdown,
)
from lessons_db.eval.analysis import (
    extract_failure_cases as extract_failure_cases,
)

# --- variants (zero internal deps) ---
from lessons_db.eval.client import (
    _clean_principle as _clean_principle,
)

# --- client (Ollama + OpenAI) ---
from lessons_db.eval.client import (
    call_judge as call_judge,
)
from lessons_db.eval.client import (
    call_ollama as call_ollama,
)
from lessons_db.eval.generate import (
    _generate_for_lesson as _generate_for_lesson,
)

# --- generate ---
from lessons_db.eval.generate import (
    run_eval_generate as run_eval_generate,
)

# --- judge ---
from lessons_db.eval.judge import (
    compute_metrics as compute_metrics,
)
from lessons_db.eval.judge import (
    compute_rank_metrics as compute_rank_metrics,
)
from lessons_db.eval.judge import (
    compute_tournament_metrics as compute_tournament_metrics,
)
from lessons_db.eval.judge import (
    parse_binary_judge as parse_binary_judge,
)
from lessons_db.eval.judge import (
    parse_judge_scores as parse_judge_scores,
)
from lessons_db.eval.judge import (
    parse_paired_judge as parse_paired_judge,
)
from lessons_db.eval.judge import (
    run_eval_judge as run_eval_judge,
)
from lessons_db.eval.judge import (
    run_paired_tournament as run_paired_tournament,
)
from lessons_db.eval.learn import (
    BEST_JSON as BEST_JSON,
)
from lessons_db.eval.learn import (
    LEARNINGS_FILE as LEARNINGS_FILE,
)
from lessons_db.eval.learn import (
    compute_ablations as compute_ablations,
)
from lessons_db.eval.learn import (
    compute_dimension_impacts as compute_dimension_impacts,
)
from lessons_db.eval.learn import (
    compute_variant_trends as compute_variant_trends,
)
from lessons_db.eval.learn import (
    derive_insights as derive_insights,
)
from lessons_db.eval.learn import (
    format_ablation_summary as format_ablation_summary,
)
from lessons_db.eval.learn import (
    load_best as load_best,
)
from lessons_db.eval.learn import (
    load_learnings as load_learnings,
)

# --- learn (post-judge insights, always-on) ---
from lessons_db.eval.learn import (
    run_eval_learn as run_eval_learn,
)
from lessons_db.eval.prompts import (
    _build_self_critique_prompt as _build_self_critique_prompt,
)

# --- prompts ---
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

# --- reports ---
from lessons_db.eval.reports import (
    compute_simulation_lift as compute_simulation_lift,
)
from lessons_db.eval.reports import (
    diagnose_vs_reference as diagnose_vs_reference,
)
from lessons_db.eval.reports import (
    parse_simulation_result as parse_simulation_result,
)
from lessons_db.eval.reports import (
    render_report as render_report,
)
from lessons_db.eval.reports import (
    render_v2_report as render_v2_report,
)
from lessons_db.eval.sampling import (
    _select_diverse as _select_diverse,
)

# --- sampling ---
from lessons_db.eval.sampling import (
    select_source_lessons as select_source_lessons,
)
from lessons_db.eval.sampling import (
    select_transfer_targets as select_transfer_targets,
)
from lessons_db.eval.signals import (
    _PRIOR_LOG_ODDS as _PRIOR_LOG_ODDS,
)

# --- signals (parsers + Bayesian fusion) ---
from lessons_db.eval.signals import (
    compute_bayesian_metrics as compute_bayesian_metrics,
)
from lessons_db.eval.signals import (
    compute_embedding_signal as compute_embedding_signal,
)
from lessons_db.eval.signals import (
    compute_mechanism_signal as compute_mechanism_signal,
)
from lessons_db.eval.signals import (
    compute_paired_signal as compute_paired_signal,
)
from lessons_db.eval.signals import (
    compute_scope_signal as compute_scope_signal,
)
from lessons_db.eval.signals import (
    compute_transfer_posterior as compute_transfer_posterior,
)
from lessons_db.eval.signals import (
    parse_mechanism_triplet as parse_mechanism_triplet,
)
from lessons_db.eval.variants import (
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
