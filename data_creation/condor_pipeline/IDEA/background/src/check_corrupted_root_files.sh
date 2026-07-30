#!/bin/bash

set -uo pipefail

readonly DEFAULT_RELEASE="2026-07-29"
readonly KEY4HEP_SETUP="/cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh"

input_directory=""
key4hep_release="$DEFAULT_RELEASE"
report="corrupted_root_files.txt"

usage() {
    cat <<EOF
Usage: $0 DIRECTORY [--release DATE] [--report FILE]

Check every .root file by reading all of its events with podio-dump.

Arguments:
  DIRECTORY        Directory containing the ROOT files to check

Options:
  --release DATE   Key4hep nightly release (default: $DEFAULT_RELEASE)
  --report FILE    File in which corrupted paths are recorded
                   (default: corrupted_root_files.txt)
  -h, --help       Show this help

Exit status: 0 if all files are valid, 1 if corrupted files are found,
and 2 if setup, input, staging, or another operational step fails.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            [[ $# -ge 2 ]] || { echo "ERROR: --release requires a value" >&2; exit 2; }
            key4hep_release=$2
            shift 2
            ;;
        --report)
            [[ $# -ge 2 ]] || { echo "ERROR: --report requires a value" >&2; exit 2; }
            report=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$input_directory" ]]; then
                echo "ERROR: only one input directory may be specified" >&2
                usage >&2
                exit 2
            fi
            input_directory=$1
            shift
            ;;
    esac
done

if [[ -z "$input_directory" ]]; then
    echo "ERROR: an input directory is required" >&2
    usage >&2
    exit 2
fi

if [[ ! -r "$KEY4HEP_SETUP" ]]; then
    echo "ERROR: Key4hep setup script is not readable: $KEY4HEP_SETUP" >&2
    exit 2
fi

# These files were produced with this nightly. Loading the same release also
# supplies matching ROOT, podio, and EDM4hep dictionaries.
set +u
# shellcheck disable=SC1090
source "$KEY4HEP_SETUP" -r "$key4hep_release"
setup_status=$?
set -u
if [[ $setup_status -ne 0 ]]; then
    echo "ERROR: failed to set up Key4hep release $key4hep_release" >&2
    exit 2
fi

if ! command -v podio-dump >/dev/null 2>&1; then
    echo "ERROR: podio-dump is not available after Key4hep setup" >&2
    exit 2
fi
if [[ ! -d "$input_directory" ]]; then
    echo "ERROR: input directory does not exist: $input_directory" >&2
    exit 2
fi

report_directory=$(dirname -- "$report")
if [[ ! -d "$report_directory" ]]; then
    echo "ERROR: report directory does not exist: $report_directory" >&2
    exit 2
fi
if ! : >"$report"; then
    echo "ERROR: cannot write report: $report" >&2
    exit 2
fi

temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/check-root-files.XXXXXX") || {
    echo "ERROR: could not create a temporary directory" >&2
    exit 2
}
cleanup() {
    rm -r -- "$temporary_directory"
}
trap cleanup EXIT

# ROOT currently crashes when these files are opened directly through the EOS
# FUSE mount. Stage one file at a time so validation is reliable and bounded in
# local disk usage. Suppress core dumps from malformed input files.
ulimit -c 0 || true

mapfile -d '' root_files < <(
    find "$input_directory" -maxdepth 1 -type f -name '*.root' -print0 | sort -z
)

if [[ ${#root_files[@]} -eq 0 ]]; then
    echo "ERROR: no .root files found in $input_directory" >&2
    exit 2
fi

valid_count=0
corrupted_count=0
unreadable_count=0
total_count=${#root_files[@]}

echo "Checking $total_count ROOT file(s) in $input_directory"

for index in "${!root_files[@]}"; do
    source_file=${root_files[$index]}
    staged_file="$temporary_directory/input.root"
    error_log="$temporary_directory/podio-dump.err"
    display_index=$((index + 1))

    printf '[%d/%d] %s: ' "$display_index" "$total_count" "$(basename -- "$source_file")"

    if ! cp -- "$source_file" "$staged_file"; then
        echo "UNREADABLE (copy failed)"
        unreadable_count=$((unreadable_count + 1))
        continue
    fi

    if podio-dump -e -1 "$staged_file" >/dev/null 2>"$error_log"; then
        echo "OK"
        valid_count=$((valid_count + 1))
    else
        echo "CORRUPTED"
        printf '%s\n' "$source_file" >>"$report"
        corrupted_count=$((corrupted_count + 1))
        sed 's/^/    /' "$error_log" >&2
    fi

    rm -f -- "$staged_file" "$error_log"
done

echo
echo "Summary: $valid_count valid, $corrupted_count corrupted, $unreadable_count unreadable"
echo "Corrupted-file report: $report"

if [[ $unreadable_count -gt 0 ]]; then
    exit 2
fi
if [[ $corrupted_count -gt 0 ]]; then
    exit 1
fi
exit 0
