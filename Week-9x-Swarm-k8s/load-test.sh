#!/bin/bash

API_URL="http://127.0.0.1:53757"

echo "Starting load test on $API_URL"
echo "Press Ctrl+C to stop the test"

MAX_JOBS=100

while true; do
  while [ "$(jobs | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 0.1
  done
  curl -X 'GET' "$API_URL/detect/stress_test" -H 'accept: application/json' > /dev/null &
  sleep 1
done


# wrk -t4 -c16 -d1m -H "accept: application/json" http://127.0.0.1:53757/detect/stress_test


# wrk -t1 -c1 -d10s -H "accept: application/json" http://0.0.0.0:8000/detect/stress_test