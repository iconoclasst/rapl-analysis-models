#!/bin/bash

OUTPUT="ae_results.txt"
PYTHON_BIN="/home/davi/consumo/venv/bin/python3"
SCRIPT="mod_ae.py"

echo "" > $OUTPUT

for opt in 1 2 3 4 5 6 7 8 9 x; do
    echo "Rodando opção $opt"
    result=$(sudo perf stat -a -e power/energy-pkg/ $PYTHON_BIN $SCRIPT <<< "$opt" 2>&1)
    joules=$(echo "$result" | grep "Joules" | awk '{print $1}')
    seconds=$(echo "$result" | grep "seconds time elapsed" | awk '{print $1}')
    echo "opt=$opt, joules=$joules, seconds=$seconds" >> $OUTPUT
done

