#!/bin/sh
# PAJEET installer. Safe to re-run. Needs python3 (>= 3.8) and curl or wget.
#
#   curl -fsSL https://raw.githubusercontent.com/YossTechLLC/PAJEET/main/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --project      # this project only
#   curl -fsSL .../install.sh | sh -s -- --dry-run      # show the settings.json diff, install nothing
#   curl -fsSL .../install.sh | sh -s -- --uninstall    # remove the hooks and the binary
#                                                       # (add --project to remove only this project's hooks,
#                                                       #  --purge to also delete the ledger)
#
# What it does: puts ONE file (pajeet) in ~/.local/bin, then runs `pajeet install`,
# which merges 5 hook entries into ~/.claude/settings.json (original kept as settings.json.pajeet.bak).
# No sudo, no packages, no network use after this download.
#
# Knobs: PAJEET_REPO (owner/name), PAJEET_REF (branch/tag/sha), PAJEET_BIN_DIR,
#        PAJEET_RAW_BASE (full base URL, wins over REPO/REF), PAJEET_SHA256 (verify the download).
set -eu

REPO="${PAJEET_REPO:-YossTechLLC/PAJEET}"
REF="${PAJEET_REF:-main}"
BASE="${PAJEET_RAW_BASE:-https://raw.githubusercontent.com/$REPO/$REF}"
BIN_DIR="${PAJEET_BIN_DIR:-${HOME:?HOME is not set}/.local/bin}"
case "$BIN_DIR" in /*) ;; *) BIN_DIR="$PWD/$BIN_DIR" ;; esac
TARGET="$BIN_DIR/pajeet"

say() { printf 'pajeet-install: %s\n' "$*"; }
die() { printf 'pajeet-install: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
usage: install.sh [--project] [--dry-run] [--uninstall] [--purge]
  (none)       install pajeet to ~/.local/bin and register hooks in ~/.claude/settings.json
  --project    register in ./.claude/settings.local.json instead (this project only)
  --dry-run    print the settings.json diff and change nothing
  --uninstall  remove the hooks and the binary  (--project: hooks only, keep the binary)
  --purge      with --uninstall: also delete the ledger in ~/.pajeet
Piped from curl: curl -fsSL <url>/install.sh | sh -s -- --dry-run
EOF
}

# Validate flags before touching anything.
mode=install
dry=0
project=0
purge=0
for a in "$@"; do
  case "$a" in
    --uninstall) mode=uninstall ;;
    --dry-run) dry=1 ;;
    --project) project=1 ;;
    --purge) purge=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $a" ;;
  esac
done

[ "$purge" = 0 ] || [ "$mode" = uninstall ] || die "--purge only makes sense with --uninstall"

command -v python3 >/dev/null 2>&1 || die "python3 not found. Install it first (WSL/Ubuntu: sudo apt install python3)."
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || die "python3 >= 3.8 is required."

if [ "$mode" = uninstall ]; then
  [ -f "$TARGET" ] || die "$TARGET not found; nothing to uninstall."
  args=""
  for a in "$@"; do [ "$a" = "--uninstall" ] || args="$args $a"; done
  # shellcheck disable=SC2086
  python3 "$TARGET" uninstall $args
  if [ "$project" = 1 ]; then
    say "kept $TARGET (other projects may still use it)"
  else
    rm -f "$TARGET" && say "removed $TARGET"
  fi
  exit 0
fi

tmp="$(mktemp "${TMPDIR:-/tmp}/pajeet.XXXXXX")" || die "cannot create a temporary file"
trap 'rm -f "$tmp"' EXIT INT TERM

# Running from a clone? Use the file next to this script. Otherwise download it.
here=""
if [ -f "$0" ]; then here="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" || here=""; fi
if [ -n "$here" ] && [ -f "$here/pajeet" ] && [ -f "$here/install.sh" ]; then
  say "using local copy: $here/pajeet"
  cp "$here/pajeet" "$tmp" || die "cannot read $here/pajeet"
else
  say "downloading $BASE/pajeet"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$BASE/pajeet" -o "$tmp" || die "download failed (is the repo public, and PAJEET_REPO / PAJEET_REF right?)"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tmp" "$BASE/pajeet" || die "download failed"
  else
    die "need curl or wget."
  fi
fi

if [ -n "${PAJEET_SHA256:-}" ]; then
  got="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$tmp")"
  [ "$got" = "$PAJEET_SHA256" ] || die "sha256 mismatch: expected $PAJEET_SHA256, got $got"
fi

# Run it through python3 (a noexec TMPDIR must not break the install) and check it says who it is.
version="$(python3 "$tmp" --version 2>/dev/null)" || die "downloaded file is not a working pajeet script."
case "$version" in "pajeet "[0-9]*) ;; *) die "downloaded file is not the pajeet script." ;; esac

if [ "$dry" = 1 ]; then
  say "dry run: would install $TARGET ($version) and change:"
  PAJEET_SELF="$TARGET" python3 "$tmp" install "$@"
  exit 0
fi

mkdir -p "$BIN_DIR" || die "cannot create $BIN_DIR"
stage="$(mktemp "$BIN_DIR/.pajeet.XXXXXX")" || die "cannot write to $BIN_DIR"
trap 'rm -f "$tmp" "$stage"' EXIT INT TERM
cp "$tmp" "$stage" || die "cannot copy into $BIN_DIR"
chmod 755 "$stage" || die "cannot chmod $stage"
mv -f "$stage" "$TARGET" || die "cannot install $TARGET"
[ "$("$TARGET" --version 2>/dev/null)" = "$version" ] || die "installed $TARGET does not run. Is python3 on PATH for /usr/bin/env?"
say "installed $TARGET ($version)"

"$TARGET" install "$@"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "note: $BIN_DIR is not on your PATH. Add this to ~/.bashrc or ~/.zshrc:"
     say "  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
