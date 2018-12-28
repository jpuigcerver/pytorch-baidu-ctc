#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

###########################################
## THIS CODE IS EXECUTED WITHIN THE HOST ##
###########################################
if [ ! -f /.dockerenv ]; then
  docker run --rm --log-driver none \
	 -v /tmp:/host/tmp \
	 -v ${SOURCE_DIR}:/host/src \
	 joapuipe/manylinux-centos7 \
	 /host/src/wheels/create_wheels_cpu.sh;
  exit 0;
fi;

#######################################################
## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER ##
#######################################################
set -ex;

# Copy host source directory, to avoid changes in the host.
cp -r /host/src /tmp/src;
cd /tmp/src;

# Install PyTorch
./wheels/install_pytorch_cpu.sh;

cd /tmp/src;
for py in cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m; do
  echo "=== Building for $py with CPU-only ==="
  export PYTHON=/opt/python/$py/bin/python;
  $PYTHON setup.py clean;
  $PYTHON setup.py bdist_wheel;
done;

# No need to fix wheels for CPU

rm -rf /opt/rh;
for py in cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m; do
  echo "=== Testing wheel for $py with CPU-only ===";
  export PYTHON=/opt/python/$py/bin/python;
  cd /tmp;
  $PYTHON -m pip uninstall -y torch_baidu_ctc;
  $PYTHON -m pip install torch_baidu_ctc --no-index -f /tmp/src/dist --no-dependencies -v;
  $PYTHON -m unittest torch_baidu_ctc.test;
  cd - 2>&1 > /dev/null;
done;

set +x;
ODIR="/host/tmp/pytorch_baidu_ctc/whl/cpu";
mkdir -p "$ODIR";
readarray -t wheels < <(find /tmp/src/dist -name "*.whl");
for whl in "${wheels[@]}"; do
  whl_name="$(basename "$whl")";
  whl_name="${whl_name/-linux/-manylinux1}";
  cp "$whl" "${ODIR}/${whl_name}";
done;

echo "================================================================";
printf "=== %-56s ===\n" "Copied ${#wheels[@]} wheels to ${ODIR:5}";
echo "================================================================";
