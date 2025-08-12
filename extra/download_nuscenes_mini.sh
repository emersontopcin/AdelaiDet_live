#!/bin/bash
# Script para baixar o nuScenes mini dataset

set -e

NUSCENES_MINI_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
OUTPUT_DIR="/root/code/nuscenes"

echo "Criando diretório de saída: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Baixando nuScenes mini dataset..."
wget -c "$NUSCENES_MINI_URL"

echo "Extraindo arquivos..."
tar -xzf v1.0-mini.tgz

echo "Download e extração concluídos."