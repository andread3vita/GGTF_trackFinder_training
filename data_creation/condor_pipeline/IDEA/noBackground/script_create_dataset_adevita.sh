#!/bin/bash

TYPE=${1}           # type: Pythia or Gun
CONFIG=${2}         # config file
VERSION=${3}        # IDEA version (2 or 3)
OPTION=${4}         # IDEA option
NFILE=${5}          # number of files
TRAIN_OR_VAL=${6}   # training or validation ('train' or 'val')
OUTDIR=${7}         # output directory

CURRPATH=$(pwd)
ORIG_PARAMS=("$@")
set --
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r 2026-04-20 # if you need to fix a specific nightly: source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh -r your_version
set -- "${ORIG_PARAMS[@]}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -d "$BASE_DIR/data_creation" && "$BASE_DIR" != "/" ]]; do
    BASE_DIR="$(dirname "$BASE_DIR")"
done
[[ "$BASE_DIR" == "/" ]] && { echo "ERROR: could not find project root containing data_creation"; exit 1; }

MAIN_DIR="$BASE_DIR/data_${TRAIN_OR_VAL}"
mkdir -p "$MAIN_DIR"

SUB_DIR="$MAIN_DIR/idea_v${VERSION}_${OPTION}_nobackground"
mkdir -p "$SUB_DIR"

outdir="$OUTDIR"

mkdir -p "$BASE_DIR/data_creation/condor_pipeline/IDEA/noBackground/gun"

if [[ "${VERSION}" -eq 3 ]]; then

    STEERING_FILE=utils/SteeringFile_IDEA_o1_v03.py

    src_file="$FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py"
    cp "$src_file" "$STEERING_FILE"
    sed -i 's/simulateCalo *= *True/simulateCalo = False/' "$STEERING_FILE"
    
fi

python src/submit_jobs.py  --queue testmatch --outdir $outdir --njobs $NFILE --type $TYPE --config $CONFIG --detectorVersion $VERSION --detectorOption $OPTION --train_or_val $TRAIN_OR_VAL