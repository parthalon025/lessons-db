# Claude Code Hooks Setup

Add these entries to `~/.claude/settings.json` to integrate lessons-db with Claude Code.

## SessionStart — status on session open

```json
{
  "matcher": "",
  "hooks": [{
    "type": "command",
    "command": "bash ~/.claude/hooks/lessons-db-session-start.sh",
    "timeout": 5
  }]
}
```

## PreToolUse:Read — surface lessons for files being read

```json
{
  "matcher": "Read",
  "hooks": [{
    "type": "command",
    "command": "bash ~/.claude/hooks/lessons-db-pre-read.sh",
    "timeout": 3
  }]
}
```

## PreToolUse:Edit — check detection patterns before edits

```json
{
  "matcher": "Edit",
  "hooks": [{
    "type": "command",
    "command": "bash ~/.claude/hooks/lessons-db-pre-edit.sh",
    "timeout": 3
  }]
}
```

## Copy hook scripts

The hook scripts are in the lessons-db repository under `hooks/`. Copy them to `~/.claude/hooks/`:

```bash
cp hooks/*.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/lessons-db-*.sh
```
