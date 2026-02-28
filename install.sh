#!/usr/bin/env bash
# install.sh — Install lessons-db globally for Claude Code
#
# Symlinks hooks, scripts, and CLI into their global locations:
#   ~/.claude/hooks/      <- hooks/*.sh
#   ~/.local/bin/         <- .venv/bin/lessons-db, .venv/bin/learn
#
# Also installs the Python package in editable mode if a venv exists.
#
# Usage:
#   ./install.sh            # install (default, skips conflicts)
#   ./install.sh --force    # install, replacing conflicts (backs up to *.bak)
#   ./install.sh --uninstall # remove all symlinks created by this script
#   ./install.sh --status    # show what's installed vs missing
set -euo pipefail

FORCE=0

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

HOOKS_DIR="${HOME}/.claude/hooks"
BIN_DIR="${HOME}/.local/bin"
VENV_DIR="${REPO_ROOT}/.venv"

# Hook scripts to install
HOOK_SCRIPTS=(
    lessons-db-enter-plan.sh
    lessons-db-post-bash.sh
    lessons-db-pre-edit.sh
    lessons-db-pre-read.sh
    lessons-db-session-start.sh
    lessons-db-stop.sh
)

# CLI binaries from the venv
CLI_BINARIES=(
    lessons-db
    learn
)

installed=0
skipped=0
replaced=0

# --- Helpers ---

log_install() { echo "  + $1"; }
log_skip()    { echo "  ~ $1 (exists, not ours — skipped)"; }
log_replace() { echo "  * $1 (updated stale symlink)"; }
log_force()   { echo "  ! $1 (backed up to .bak, replaced)"; }
log_remove()  { echo "  - $1"; }

symlink_item() {
    local src="$1" dest="$2" label="$3"

    if [[ -L "$dest" ]]; then
        local current target
        current=$(readlink -f "$dest" 2>/dev/null || true)
        target=$(readlink -f "$src" 2>/dev/null || true)
        if [[ "$current" == "$target" ]]; then
            return 0  # already correct
        fi
        ln -sfn "$src" "$dest"
        log_replace "$label"
        replaced=$((replaced + 1))
    elif [[ -e "$dest" ]]; then
        if [[ "$FORCE" -eq 1 ]]; then
            mv "$dest" "${dest}.bak"
            ln -s "$src" "$dest"
            log_force "$label"
            replaced=$((replaced + 1))
        else
            log_skip "$label"
            skipped=$((skipped + 1))
        fi
    else
        ln -s "$src" "$dest"
        log_install "$label"
        installed=$((installed + 1))
    fi
}

remove_if_ours() {
    local src="$1" dest="$2" label="$3"
    if [[ -L "$dest" ]]; then
        local current target
        current=$(readlink -f "$dest" 2>/dev/null || true)
        target=$(readlink -f "$src" 2>/dev/null || true)
        if [[ "$current" == "$target" ]]; then
            rm "$dest"
            log_remove "$label"
            installed=$((installed + 1))
        fi
    fi
}

# --- Install ---

do_install() {
    echo "Installing lessons-db..."
    echo ""

    mkdir -p "$HOOKS_DIR" "$BIN_DIR"

    # Install Python package in editable mode if venv exists
    if [[ -d "$VENV_DIR" ]]; then
        echo "Python package:"
        if "$VENV_DIR/bin/python" -m pip show lessons-db &>/dev/null; then
            echo "  ~ lessons-db package already installed"
        else
            echo "  + Installing lessons-db in editable mode..."
            "$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT" -q
        fi
        echo ""
    else
        echo "WARNING: No .venv found at $VENV_DIR — skipping Python package install"
        echo "  Run: python3 -m venv .venv && .venv/bin/python -m pip install -e ."
        echo ""
    fi

    echo "Hooks:"
    for hook in "${HOOK_SCRIPTS[@]}"; do
        src="${REPO_ROOT}/hooks/${hook}"
        if [[ -f "$src" ]]; then
            symlink_item "$src" "$HOOKS_DIR/$hook" "hooks/$hook"
        fi
    done

    echo ""
    echo "CLI (~/.local/bin):"
    for bin in "${CLI_BINARIES[@]}"; do
        src="${VENV_DIR}/bin/${bin}"
        if [[ -f "$src" ]]; then
            symlink_item "$src" "$BIN_DIR/$bin" "bin/$bin"
        else
            echo "  ~ bin/$bin not found in venv (install package first)"
        fi
    done

    echo ""
    echo "Done: $installed installed, $replaced updated, $skipped skipped"
}

# --- Uninstall ---

do_uninstall() {
    installed=0
    echo "Uninstalling lessons-db..."
    echo ""

    echo "Hooks:"
    for hook in "${HOOK_SCRIPTS[@]}"; do
        src="${REPO_ROOT}/hooks/${hook}"
        if [[ -f "$src" ]]; then
            remove_if_ours "$src" "$HOOKS_DIR/$hook" "hooks/$hook"
        fi
    done

    echo ""
    echo "CLI:"
    for bin in "${CLI_BINARIES[@]}"; do
        src="${VENV_DIR}/bin/${bin}"
        if [[ -f "$src" ]]; then
            remove_if_ours "$src" "$BIN_DIR/$bin" "bin/$bin"
        fi
    done

    echo ""
    echo "Removed $installed symlinks"
}

# --- Status ---

do_status() {
    echo "lessons-db install status"
    echo "Repo: $REPO_ROOT"
    echo ""

    local ok=0 missing=0 stale=0 conflict=0

    check_item() {
        local src="$1" dest="$2" label="$3"
        if [[ -L "$dest" ]]; then
            local current target
            current=$(readlink -f "$dest" 2>/dev/null || true)
            target=$(readlink -f "$src" 2>/dev/null || true)
            if [[ "$current" == "$target" ]]; then
                ok=$((ok + 1))
            else
                echo "  STALE    $label -> $(readlink "$dest")"
                stale=$((stale + 1))
            fi
        elif [[ -e "$dest" ]]; then
            echo "  CONFLICT $label (exists, not a symlink)"
            conflict=$((conflict + 1))
        else
            echo "  MISSING  $label"
            missing=$((missing + 1))
        fi
    }

    # Python package
    echo "Python package:"
    if [[ -d "$VENV_DIR" ]] && "$VENV_DIR/bin/python" -m pip show lessons-db &>/dev/null; then
        local location
        location=$("$VENV_DIR/bin/python" -m pip show lessons-db 2>/dev/null | grep "^Location:" | cut -d' ' -f2)
        echo "  OK  lessons-db ($location)"
    else
        echo "  MISSING  lessons-db package"
    fi

    echo ""
    echo "Hooks:"
    for hook in "${HOOK_SCRIPTS[@]}"; do
        src="${REPO_ROOT}/hooks/${hook}"
        if [[ -f "$src" ]]; then
            check_item "$src" "$HOOKS_DIR/$hook" "hooks/$hook"
        fi
    done

    echo ""
    echo "CLI:"
    for bin in "${CLI_BINARIES[@]}"; do
        src="${VENV_DIR}/bin/${bin}"
        if [[ -f "$src" ]]; then
            check_item "$src" "$BIN_DIR/$bin" "bin/$bin"
        fi
    done

    echo ""
    echo "OK: $ok  Missing: $missing  Stale: $stale  Conflict: $conflict"
}

# --- Main ---

case "${1:-}" in
    --uninstall) do_uninstall ;;
    --status)    do_status ;;
    --force)     FORCE=1; do_install ;;
    *)           do_install ;;
esac
