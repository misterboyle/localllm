#!/usr/bin/env bash
set -euo pipefail

# swap-model.sh — Swap the active model variant for a server
# Updates both models.jsonc and opencode.jsonc, then tells you to restart.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVERS_CONFIG="$HOME/.localllm/models.jsonc"
OPENCODE_CONFIG="$HOME/.config/opencode/opencode.jsonc"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

usage() {
  echo "Usage: $0 <server> <model-variant>"
  echo ""
  echo "Swap the model for a server in both models.jsonc and opencode.jsonc."
  echo ""
  echo "Servers: dense, moe"
  echo ""
  echo "Examples:"
  echo "  $0 dense Qwen3.6-27B-8bit"
  echo "  $0 dense Qwen3.6-27B-UD-MLX-4bit"
  echo ""
  echo "Available variants (from models.jsonc):"
  list_variants
  exit 0
}

list_variants() {
  python3 -c "
import json, sys

def strip_jsonc(text):
    result = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            result.append(c)
            escape = False
            i += 1
            continue
        if in_string:
            result.append(c)
            if c == '\\\\':
                escape = True
            elif c == '\"':
                in_string = False
            i += 1
            continue
        if c == '\"':
            in_string = True
            result.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < len(text) and text[i + 1] == '/':
            while i < len(text) and text[i] != '\\n':
                i += 1
            continue
        result.append(c)
        i += 1
    return ''.join(result)

try:
    raw = open('$SERVERS_CONFIG').read()
    raw = strip_jsonc(raw)
    cfg = json.loads(raw)
    variants = cfg.get('modelVariants', {})
    if not variants:
        print('  (no modelVariants section found in models.jsonc)')
        sys.exit(0)
    for server, models in variants.items():
        print(f'  {server}:')
        for m in models:
            active = cfg.get('servers', {}).get(server, {}).get('model', '')
            marker = ' <-- active' if m == active else ''
            print(f'    {m}{marker}')
except FileNotFoundError:
    print(f'  Config not found: $SERVERS_CONFIG')
    sys.exit(1)
"
}

# Validate args
if [ $# -lt 2 ]; then
  if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    usage
  fi
  echo "ERROR: missing arguments" >&2
  usage
fi

SERVER="$1"
MODEL="$2"

# Validate server name
case "$SERVER" in
  dense|moe) ;;
  *) echo "ERROR: unknown server '$SERVER'. Use: dense, moe" >&2; exit 1 ;;
esac

# Validate model directory exists
MODEL_PATH="$HOME/.localllm/models/$MODEL"
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model directory not found: $MODEL_PATH" >&2
  echo "Available models:" >&2
  ls -1 "$HOME/.localllm/models/" >&2
  exit 1
fi

# Check current model
CURRENT=$(python3 -c "
import json, sys

def strip_jsonc(text):
    result = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            result.append(c)
            escape = False
            i += 1
            continue
        if in_string:
            result.append(c)
            if c == '\\\\':
                escape = True
            elif c == '\"':
                in_string = False
            i += 1
            continue
        if c == '\"':
            in_string = True
            result.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < len(text) and text[i + 1] == '/':
            while i < len(text) and text[i] != '\\n':
                i += 1
            continue
        result.append(c)
        i += 1
    return ''.join(result)

raw = open('$SERVERS_CONFIG').read()
raw = strip_jsonc(raw)
cfg = json.loads(raw)
print(cfg.get('servers', {}).get('$SERVER', {}).get('model', ''))
")

if [ "$CURRENT" = "$MODEL" ]; then
  log "Already active: $SERVER = $MODEL"
  exit 0
fi

log "Swapping $SERVER: $CURRENT -> $MODEL"

# Update models.jsonc — preserve comments via targeted string replacement
python3 -c "
import json, re, sys

def strip_jsonc(text):
    result = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            result.append(c)
            escape = False
            i += 1
            continue
        if in_string:
            result.append(c)
            if c == '\\\\':
                escape = True
            elif c == '\"':
                in_string = False
            i += 1
            continue
        if c == '\"':
            in_string = True
            result.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < len(text) and text[i + 1] == '/':
            while i < len(text) and text[i] != '\\n':
                i += 1
            continue
        result.append(c)
        i += 1
    return ''.join(result)

config_path = '$SERVERS_CONFIG'
raw = open(config_path).read()
raw_stripped = strip_jsonc(raw)
cfg = json.loads(raw_stripped)

# Update the model in the parsed config
cfg['servers']['$SERVER']['model'] = '$MODEL'

# Write back preserving comments: find the server block and replace the model value
result = raw
server_start = raw.find('\"$SERVER\"')
if server_start == -1:
    print('ERROR: server $SERVER not found', file=sys.stderr)
    sys.exit(1)

# Find the model line within this server block
block = raw[server_start:server_start + 500]
model_match = re.search(r'(\"model\":\s*\")[^\"]+\"', block)
if model_match:
    full_match = model_match.group(0)
    # Extract current value between quotes
    value_match = re.search(r'\"model\":\s*\"([^\"]+)\"', full_match)
    if value_match:
        old_value = value_match.group(1)
        new_value = '$MODEL'
        replacement = full_match.replace(old_value, new_value)
        result = raw[:server_start] + raw[server_start:].replace(full_match, replacement, 1)
    else:
        print('ERROR: could not extract model value', file=sys.stderr)
        sys.exit(1)
else:
    print('ERROR: could not find model field in server block', file=sys.stderr)
    sys.exit(1)

with open(config_path, 'w') as f:
    f.write(result)
"

# Update opencode.jsonc — add model entry if not present
if [ -f "$OPENCODE_CONFIG" ]; then
  python3 -c "
import json, re, os, sys

def strip_jsonc(text):
    result = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            result.append(c)
            escape = False
            i += 1
            continue
        if in_string:
            result.append(c)
            if c == '\\\\':
                escape = True
            elif c == '\"':
                in_string = False
            i += 1
            continue
        if c == '\"':
            in_string = True
            result.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < len(text) and text[i + 1] == '/':
            while i < len(text) and text[i] != '\\n':
                i += 1
            continue
        result.append(c)
        i += 1
    return ''.join(result)

config_path = '$OPENCODE_CONFIG'
raw = open(config_path).read()
raw_stripped = strip_jsonc(raw)
cfg = json.loads(raw_stripped)

provider_name = 'mlx-$SERVER'
model_path = os.path.expanduser('$HOME/.localllm/models/$MODEL')

if provider_name not in cfg.get('provider', {}):
    print(f'WARNING: provider {provider_name} not found in opencode config, skipping opencode update')
else:
    if model_path not in cfg['provider'][provider_name].get('models', {}):
        cfg['provider'][provider_name]['models'][model_path] = {
            'name': '$MODEL',
            'maxTokens': 262144,
            'limit': {
                'context': 262144,
                'output': 8192
            }
        }
        # Write back preserving comments by appending the new entry
        # Find the models section for this provider and insert before closing brace
        provider_start = raw.find('\"' + provider_name + '\"')
        if provider_start != -1:
            # Find the models key within this provider block
            block = raw[provider_start:provider_start + 2000]
            models_match = re.search(r'(\"models\":\s*\{)', block)
            if models_match:
                # Find the closing brace for this models object
                models_start = provider_start + models_match.start()
                brace_count = 0
                models_end = models_start
                for i in range(models_start, min(models_start + 2000, len(raw))):
                    if raw[i] == '{':
                        brace_count += 1
                    elif raw[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            models_end = i
                            break

                # Build the new model entry
                indent = '        '
                new_entry = (
                    f'\n{indent}// Alternative variant — use when {MODEL} is loaded in models.jsonc\n'
                    f'{indent}\"{model_path}\": {{\n'
                    f'{indent}  \"name\": \"{MODEL}\",\n'
                    f'{indent}  \"maxTokens\": 262144,\n'
                    f'{indent}  \"limit\": {{\n'
                    f'{indent}    \"context\": 262144,\n'
                    f'{indent}    \"output\": 8192\n'
                    f'{indent}  }}\n'
                    f'{indent}}}'
                )
                result = raw[:models_end] + new_entry + raw[models_end:]
                with open(config_path, 'w') as f:
                    f.write(result)
                print(f'Added model entry to opencode config: {model_path}')
            else:
                print(f'WARNING: could not find models section for {provider_name}, skipping')
        else:
            print(f'WARNING: could not find provider {provider_name} in config, skipping')
    else:
        print(f'Model entry already in opencode config: {model_path}')
"
fi

# Update agent model paths in agent config files
AGENT_DIR="$HOME/.config/opencode/agent"
if [ -d "$AGENT_DIR" ]; then
  OLD_MODEL_PATH="$HOME/.localllm/models/$CURRENT"
  NEW_MODEL_PATH="$HOME/.localllm/models/$MODEL"
  UPDATED=0
  for agent_file in "$AGENT_DIR"/*.md; do
    if [ -f "$agent_file" ] && grep -q "$OLD_MODEL_PATH" "$agent_file"; then
      sed -i '' "s|$OLD_MODEL_PATH|$NEW_MODEL_PATH|" "$agent_file"
      log "Updated agent: $(basename "$agent_file")"
      UPDATED=$((UPDATED + 1))
    fi
  done
  if [ "$UPDATED" -eq 0 ]; then
    log "No agent files needed updating."
  fi
fi

log "Done. Restart the server to apply:"
log "  ./start-server.sh stop && ./start-server.sh $SERVER"
