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

ODIR="/host/tmp/pytorch_baidu_ctc/whl/cpu";
mkdir -p "$ODIR";
wheels=();
for py in cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m; do
  export PYTHON=/opt/python/$py/bin/python;
  cd /tmp/src;
  # Remove previous builds.
  rm -rf build dist;

  "$PYTHON" -m pip install -U pip;
  "$PYTHON" -m pip install -U wheel setuptools;

  echo "=== Installing requirements for $py with CPU-only ==="
  ./wheels/install_pytorch_cpu.sh "$py";
  "$PYTHON" -m pip install \
	    -r <(sed -r 's|^torch((>=\|>).*)?$||g;/^$/d' requirements.txt);

  echo "=== Building wheel for $py with CPU-only ==="
  $PYTHON setup.py clean;
  $PYTHON setup.py bdist_wheel;

  # No need to fix wheel for CPU

  # Move dev libraries to a different location to make sure that tests do
  # not use them.
  mv /opt/rh /opt/rh_tmp;

  echo "=== Installing wheel for $py with CPU-only ==="
  cd /tmp;
  $PYTHON -m pip uninstall -y torch_baidu_ctc;
  $PYTHON -m pip install torch_baidu_ctc --no-index -f /tmp/src/dist \
	  --no-dependencies -v;

  echo "=== Testing wheel for $py with CPU-only ===";
  $PYTHON -m unittest torch_baidu_ctc.test;

  # Move dev libraries back to their original location after tests.
  mv /opt/rh_tmp /opt/rh;

  echo "=== Copying wheel for $py with CPU-only to the host ===";
  readarray -t whl < <(find /tmp/src/dist -name "*.whl");
  whl_name="$(basename "$whl")";
  whl_name="${whl_name/-linux/-manylinux1}";
  mv "$whl" "${ODIR}/${whl_name}";
  wheels+=("${ODIR}/${whl_name}");
done;

echo "================================================================";
printf "=== %-56s ===\n" "Copied ${#wheels[@]} wheels to ${ODIR:5}";
echo "================================================================";
