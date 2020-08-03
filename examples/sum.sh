#/bin/bash

tail -5 log.csv
awk 'END {print NR}' log.csv
