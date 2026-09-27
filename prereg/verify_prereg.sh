#!/usr/bin/env sh
# Verify every frozen protocol text against the SHA-256 posted before its results existed.
cd "$(dirname "$0")" || exit 1
fail=0
for f in $(python3 -c 'import json;print(" ".join(json.load(open("HASHES.json"))))'); do
  want=$(python3 -c "import json;print(json.load(open('HASHES.json'))['$f']['sha256'])")
  got=$( (sha256sum "$f" 2>/dev/null || shasum -a 256 "$f") | cut -d' ' -f1)
  if [ "$want" = "$got" ]; then echo "OK   $f"; else echo "FAIL $f"; fail=1; fi
done
exit $fail
