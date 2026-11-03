#!/usr/bin/env bash
# Python arka ucunu (src/predict.py) PyInstaller ile tek klasörlük bir programa dönüştürür.
# Çıktı: build/backend/predict-backend/ — Electron paketine (dmg/exe) extraResources olarak girer.
# Kullanım: scripts/build_backend.sh   (PYTHON=python ile başka bir yorumlayıcı seçilebilir)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
"$PY" -m PyInstaller --noconfirm --clean --log-level WARN \
  --name predict-backend --paths src \
  --distpath build/backend --workpath build/pyinstaller --specpath build \
  --collect-all torchvision \
  --exclude-module matplotlib --exclude-module sklearn --exclude-module scipy --exclude-module tkinter \
  src/predict.py

# Duman testi: paketlenmiş arka uç modeli yükleyip hazır olduğunu bildirmeli
exe=build/backend/predict-backend/predict-backend
[ -f "$exe.exe" ] && exe="$exe.exe"
# set -e altında program hata verirse çıktısı kaybolmasın diye önce yazdırılıyor
out=$(echo "" | "$exe" --serve --model models/mnist_cnn.pt) || { echo "$out"; echo "Arka uç çalışmadı" >&2; exit 1; }
echo "$out"
echo "$out" | grep -q '^@@READY' || { echo "Arka uç hazır olmadı" >&2; exit 1; }
