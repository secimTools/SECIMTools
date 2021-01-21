#!/bin/bash

TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

for i in $(ls ${SCRIPT_DIR}/ST* ${SCRIPT_DIR}/TEST*); do
    echo -e "Running ${i}:\n"
    sh ${i}
done
