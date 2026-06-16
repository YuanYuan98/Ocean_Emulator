#!/usr/bin/env bash
set -euo pipefail

codex_home="${CODEX_HOME:-/pscratch/sd/y/yy6080/codex}"
codex_bin="${CODEX_BIN:-/pscratch/sd/y/yy6080/bin/codex}"
real_bin="${CODEX_REAL_BIN:-${codex_bin}.real}"
config_file="${codex_home}/config.toml"

mkdir -p "$codex_home" "$(dirname "$codex_bin")"

if [[ ! -e "$real_bin" ]]; then
  if [[ ! -x "$codex_bin" ]]; then
    echo "codex binary not found or not executable: $codex_bin" >&2
    exit 1
  fi
  mv "$codex_bin" "$real_bin"
fi

if [[ ! -x "$real_bin" ]]; then
  echo "real codex binary not executable: $real_bin" >&2
  exit 1
fi

cat > "$codex_bin" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail

real="${CODEX_REAL_BIN:-/pscratch/sd/y/yy6080/bin/codex.real}"

if [[ ! -x "$real" ]]; then
  echo "codex wrapper: real binary not executable: $real" >&2
  exit 127
fi

if [[ "${CODEX_FORCE_FULL_ACCESS:-1}" == "0" ]]; then
  exec "$real" "$@"
fi

for arg in "$@"; do
  case "$arg" in
    --dangerously-bypass-approvals-and-sandbox)
      exec "$real" "$@"
      ;;
  esac
done

exec "$real" \
  --dangerously-bypass-approvals-and-sandbox \
  --sandbox danger-full-access \
  --ask-for-approval never \
  "$@"
WRAPPER

chmod +x "$codex_bin"

touch "$config_file"
tmp_config="$(mktemp)"
awk '
  BEGIN {
    seen_approval = 0
    seen_sandbox = 0
    inserted = 0
  }
  /^[[:space:]]*approval_policy[[:space:]]*=/ {
    print "approval_policy = \"never\""
    seen_approval = 1
    next
  }
  /^[[:space:]]*sandbox_mode[[:space:]]*=/ {
    print "sandbox_mode = \"danger-full-access\""
    seen_sandbox = 1
    next
  }
  /^[[:space:]]*\[/ && inserted == 0 {
    if (seen_approval == 0) print "approval_policy = \"never\""
    if (seen_sandbox == 0) print "sandbox_mode = \"danger-full-access\""
    inserted = 1
  }
  { print }
  END {
    if (inserted == 0) {
      if (seen_approval == 0) print "approval_policy = \"never\""
      if (seen_sandbox == 0) print "sandbox_mode = \"danger-full-access\""
    }
  }
' "$config_file" > "$tmp_config"
mv "$tmp_config" "$config_file"

echo "Installed codex full-access wrapper:"
echo "  wrapper: $codex_bin"
echo "  real:    $real_bin"
echo "  config:  $config_file"
"$codex_bin" --version
