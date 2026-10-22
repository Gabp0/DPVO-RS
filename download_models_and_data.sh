#!/bin/bash
set -e

URL="https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AGHwGnUAua9-jQvD9ihTyjA?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1"
TMP=dpvo_download

mkdir -p "$TMP" datasets
wget "$URL" -O "$TMP/folder.zip"

if file "$TMP/folder.zip" | grep -qi html; then
    echo "Error: got an HTML page instead of the folder zip." >&2
    exit 1
fi

# The folder zip may give a "mismatching local filename" warning with exit code 1
unzip -o "$TMP/folder.zip" -d "$TMP" || true

find_file() {
    local f
    f=$(find "$TMP" -name "$1" -not -path "$TMP/folder.zip" | head -n 1)
    if [ -z "$f" ]; then
        echo "Error: $1 not found in the folder." >&2
        exit 1
    fi
    echo "$f"
}

mv "$(find_file TartanAir.pickle)" datasets/
unzip -o "$(find_file models.zip)"
unzip -o "$(find_file movies.zip)"

rm -rf "$TMP"