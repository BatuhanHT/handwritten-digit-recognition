#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Python 3.10+ gerekir; macOS'un sistem python3'ü 3.9 olabilir.
PY=""
for cand in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 veya üstü bulunamadı. macOS'ta: brew install python@3.12" >&2
  exit 1
fi
echo "Kullanılan Python: $("$PY" --version)"

# --clear: eski (ör. 3.9 ile kurulmuş) bir .venv varsa baştan kurar
"$PY" -m venv --clear .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Kurulum tamam. Etkinleştirmek için: source .venv/bin/activate"
