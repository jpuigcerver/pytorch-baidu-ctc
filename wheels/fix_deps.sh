#!/bin/bash
set -ex;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";

function fname_with_sha256() {
  HASH=$(sha256sum $1 | cut -c1-8)
  DIRNAME=$(dirname $1)
  BASENAME=$(basename $1)
  if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
    echo $1
  else
    INITNAME=$(echo $BASENAME | cut -f1 -d".")
    ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
    echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
  fi
}

function make_wheel_record() {
  FPATH=$1
  if echo $FPATH | grep RECORD >/dev/null 2>&1; then
    # if the RECORD file, then
    echo "$FPATH,,"
  else
    HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
    FSIZE=$(ls -nl $FPATH | awk '{print $5}')
    echo "$FPATH,sha256=$HASH,$FSIZE"
  fi
}

DIST_DIR="$1";
PACKAGE="$2";
shift 2;
DEPS_SONAME=();
DEPS_LIST=();
while [ $# -ge 2 ]; do
  DEPS_SONAME+=("$1");
  DEPS_LIST+=("$2");
  shift 2;
done;
[ $# -ne 0 ] &&
echo "ERROR: You must specify pairs of library name and filepath" >&2 &&
exit 1;

cd "$DIST_DIR";
for pkg in "$PWD/"*.whl; do
  [ -f "$pkg" ] || continue;
  rm -rf tmp;
  mkdir -p tmp;
  cd tmp;
  cp "$pkg" .;

  unzip -q "$(basename "$pkg")";
  rm -f "$(basename "$pkg")";

  patched=();
  for filepath in "${DEPS_LIST[@]}"; do
    destpath="$PACKAGE/$(basename "$filepath")";
    [[ "$filepath" != "$destpath" ]] && cp "$filepath" "$destpath";
    patchedpath="$(fname_with_sha256 $destpath)";
    [[ "$destpath" != "$patchedpath" ]] && mv "$destpath" "$patchedpath";
    patchedname="$(basename $patchedpath)";
    patched+=("$patchedname");
    echo "Copied $filepath to $patchedpath";
  done;

  for i in $(seq ${#DEPS_LIST[@]}); do
    origname="${DEPS_SONAME[i - 1]}";
    patchedname="${patched[i - 1]}";
    [ "$origname" = "$patchedname" ] && continue;

    find "$PACKAGE/" -name "*.so*" |
    while read sofile; do
      if ( patchelf --print-needed "$sofile" |
	     grep "$origname" 2>&1 > /dev/null ); then
	echo "patching $sofile entry $origname to $patchedname";
	patchelf --replace-needed "$origname" "$patchedname" "$sofile";
      fi;
    done;
  done;

  # set RPATHs
  find "$PACKAGE" -maxdepth 1 -type f -name "*.so" |
  while read sofile; do
    echo "Setting rpath of $sofile to" '$ORIGIN:$ORIGIN/lib';
    patchelf --set-rpath '$ORIGIN' "$sofile"
    patchelf --print-rpath "$sofile";
  done;

  # regenerate the RECORD file with new hashes
  record_file="$(echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g')";
  if [[ -e "$record_file" ]]; then
    echo "Generating new record file $record_file";
    rm -f "$record_file";
    # generate records for torch folder
    find "$PACKAGE" -type f | while read fname; do
      echo "$(make_wheel_record $fname)" >> "$record_file";
    done;
    # generate records for torch-[version]-dist-info folder
    find "$PACKAGE"*dist-info -type f | while read fname; do
      echo "$(make_wheel_record $fname)" >> "$record_file";
    done;
  fi;

  zip -rq "$(basename "$pkg")" "$PACKAGE"*;
  rm -f "$pkg";
  mv "$(basename "$pkg")" "$pkg";
  cd ..;
  rm -rf tmp;
done;
