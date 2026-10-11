#!/usr/bin/env bash
# Pan-tilt firmware: build / upload / monitor, through a local arduino-cli (nothing global).
set -euo pipefail
J="$(cd "$(dirname "$0")" && pwd)"
export ARDUINO_DIRECTORIES_DATA="$J/.arduino/data"
export ARDUINO_DIRECTORIES_DOWNLOADS="$J/.arduino/staging"
export ARDUINO_DIRECTORIES_USER="$J/.arduino/user"
export ARDUINO_BUILD_CACHE_PATH="$J/.arduino/cache"
CLI=("$J/.tools/arduino-cli" --config-file "$J/arduino-cli.yaml")
FQBN=esp8266:esp8266:d1_mini
PORT=${PORT:-/dev/ttyUSB0}
SKETCH="$J/firmware/pantilt"

case "${1:-}" in
  build)   "${CLI[@]}" compile --fqbn "$FQBN" "$SKETCH" ;;
  upload)  "${CLI[@]}" compile --fqbn "$FQBN" "$SKETCH" && "${CLI[@]}" upload --fqbn "$FQBN" -p "$PORT" "$SKETCH" ;;
  monitor) "${CLI[@]}" monitor -p "$PORT" -c baudrate=115200 ;;
  *)       echo "usage: $0 build|upload|monitor" >&2; exit 1 ;;
esac
