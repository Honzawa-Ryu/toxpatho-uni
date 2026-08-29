#!/bin/bash
# Open TG-GATEs の公開メタデータ（CSV）を data/ に取得する。
# パッチ実体は /work/gd43/share/tggates/sample_patch_agg にある前提（本スクリプトでは取得しない）。
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"
cd "$DEST"

BASE_MAIN=https://dbarchive.biosciencedbc.jp/data/open-tggates/LATEST
BASE_IMG=https://dbarchive.biosciencedbc.jp/data/open-tggates-pathological-images/LATEST

# 所見(FINDING_TYPE/GRADE_TYPE)、個体メタ(化合物/用量/屠殺時点)、画像メタ、化合物一覧
for f in open_tggates_pathology open_tggates_individual open_tggates_cel_file_attribute open_tggates_main; do
    echo "fetching $f.zip"
    curl -sSLf -O "$BASE_MAIN/$f.zip"
done
echo "fetching open_tggates_pathological_image.zip"
curl -sSLf -O "$BASE_IMG/open_tggates_pathological_image.zip"

for z in *.zip; do
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$z"
done

echo
echo "done. CSV は cp932 エンコーディングであることに注意（utf-8 で読むと落ちる）。"
ls -la ./*.csv
