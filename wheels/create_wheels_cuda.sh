#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

###########################################
## THIS CODE IS EXECUTED WITHIN THE HOST ##
###########################################

if [ ! -f /.dockerenv ]; then
  DOCKER_IMAGES=(
    soumith/manylinux-cuda80
    soumith/manylinux-cuda90
    soumith/manylinux-cuda92
  );
  for image in "${DOCKER_IMAGES[@]}"; do
    docker run --runtime=nvidia --rm --log-driver none \
	   -v /tmp:/host/tmp \
	   -v ${SOURCE_DIR}:/host/src \
	   "$image" \
	   /host/src/wheels/create_wheels_cuda.sh;
  done;
  exit 0;
fi;

#######################################################
## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER ##
#######################################################
set -ex;

# Install zip, apparently is not installed.
yum install -y zip openssl;

# Copy host source directory, to avoid changes in the host.
cp -r /host/src /tmp/src;
cd /tmp/src;
rm -rf dist build;

# Detect CUDA version
export CUDA_VERSION=$(nvcc --version|tail -n1|cut -f5 -d" "|cut -f1 -d",");
export CUDA_VERSION_S="cu$(echo $CUDA_VERSION | tr -d .)";
echo "CUDA $CUDA_VERSION Detected";

# See https://github.com/baidu-research/warp-ctc/blob/master/CMakeLists.txt#L31
export CUDA_ARCH_LIST="3.5;5.0+PTX;5.2;6.0;6.1;6.2";

# Install PyTorch
./wheels/install_pytorch_cuda.sh;

for py in cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m; do
  echo "=== Building wheel for $py with CUDA ${CUDA_VERSION} ===";
  export PYTHON=/opt/python/$py/bin/python;
  $PYTHON setup.py clean;
  $PYTHON setup.py bdist_wheel;
done;

echo "=== Fixing wheels with CUDA ${CUDA_VERSION} ===";
wheels/fix_deps.sh \
  dist torch_baidu_ctc \
  "libcudart.so.${CUDA_VERSION}" \
  "/usr/local/cuda-${CUDA_VERSION}/lib64/libcudart.so.${CUDA_VERSION}";

rm -rf /opt/rh /usr/local/cuda*;
for py in cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m; do
  echo "=== Testing wheel for $py with CUDA ${CUDA_VERSION} ===";
  export PYTHON=/opt/python/$py/bin/python;
  cd /tmp;
  $PYTHON -m pip uninstall -y torch_baidu_ctc;
  $PYTHON -m pip install torch_baidu_ctc --no-index -f /tmp/src/dist --no-dependencies -v;
  $PYTHON -m unittest torch_baidu_ctc.test;
  cd - 2>&1 > /dev/null;
done;

set +x;
ODIR="/host/tmp/pytorch_baidu_ctc/whl/${CUDA_VERSION_S}";
mkdir -p "$ODIR";
cp /tmp/src/dist/*.whl "$ODIR/";
echo "================================================================";
printf "=== %-56s ===\n" "Copied wheels to ${ODIR:5}";
echo "================================================================";
