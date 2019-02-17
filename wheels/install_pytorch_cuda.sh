#!/bin/bash
set -e;

[ $# -ne 1 ] && { echo "Missing python version!" >&2 && exit 1; }

BASE_URL="http://download.pytorch.org/whl/${CUDA_VERSION_S}";
case "$1" in
  cp27-cp27mu)
    URL="${BASE_URL}/torch-1.0.1.post2-cp27-cp27mu-linux_x86_64.whl";
    ;;
  cp35-cp35m)
    URL="${BASE_URL}/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl";
    ;;
  cp36-cp36m)
    URL="${BASE_URL}/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl";
    ;;
  cp37-cp37m)
    URL="${BASE_URL}/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl";
    ;;
  *)
    echo "Unsupported Python version: $0" >&2 && exit 1;
esac;

"$PYTHON" -m pip install "$URL";
