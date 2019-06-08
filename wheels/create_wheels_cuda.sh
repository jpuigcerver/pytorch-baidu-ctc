#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

###########################################
## THIS CODE IS EXECUTED WITHIN THE HOST ##
###########################################
if [ ! -f /.dockerenv ]; then
  DOCKER_IMAGES=(
    # soumith/manylinux-cuda80  # Note: Not supported by PyTorch anymore.
    soumith/manylinux-cuda90
    soumith/manylinux-cuda100
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

# Detect CUDA version
export CUDA_VERSION=$(nvcc --version|tail -n1|cut -f5 -d" "|cut -f1 -d",");
export CUDA_VERSION_S="cu$(echo $CUDA_VERSION | tr -d .)";
echo "CUDA $CUDA_VERSION Detected";

# See https://github.com/baidu-research/warp-ctc/blob/master/CMakeLists.txt#L31
export CUDA_ARCH_LIST="3.5;5.0+PTX;5.2;6.0;6.1;6.2";

ODIR="/host/tmp/pytorch_baidu_ctc/whl/${CUDA_VERSION_S}";
mkdir -p "$ODIR";
wheels=();
for py in cp27-cp27mu cp35-cp35m cp36-cp36m cp37-cp37m; do
  export PYTHON=/opt/python/$py/bin/python;
  cd /tmp/src;
  # Remove previous builds.
  rm -rf build dist;

  "$PYTHON" -m pip install -U pip;
  "$PYTHON" -m pip install -U wheel setuptools;

  echo "=== Installing requirements for $py with CUDA ${CUDA_VERSION} ===";
  ./wheels/install_pytorch_cuda.sh "$py";
  "$PYTHON" -m pip install \
	    -r <(sed -r 's|^torch((>=\|>).*)?$||g;/^$/d' requirements.txt);

  echo "=== Building wheel for $py with CUDA ${CUDA_VERSION} ===";
  $PYTHON setup.py clean;
  $PYTHON setup.py bdist_wheel;

  echo "=== Fixing wheel for $py with CUDA ${CUDA_VERSION} ===";
  wheels/fix_deps.sh \
    dist torch_baidu_ctc \
    "libcudart.so.${CUDA_VERSION}" \
    "/usr/local/cuda-${CUDA_VERSION}/lib64/libcudart.so.${CUDA_VERSION}";

  # Move dev libraries to a different location to make sure that tests do
  # not use them.
  mv /opt/rh /opt/rh_tmp;
  for cuda_dir in /usr/local/cuda*; do
    mv "$cuda_dir" "${cuda_dir}_tmp";
  done;

  echo "=== Installing wheel for $py with CUDA ${CUDA_VERSION} ===";
  cd /tmp;
  $PYTHON -m pip uninstall -y torch_baidu_ctc;
  $PYTHON -m pip install torch_baidu_ctc --no-index -f /tmp/src/dist \
	  --no-dependencies -v;

  echo "=== Testing wheel for $py with CUDA ${CUDA_VERSION} ===";
  $PYTHON -m unittest torch_baidu_ctc.test;

  # Move dev libraries back to their original location after tests.
  mv /opt/rh_tmp /opt/rh;
  for cuda_dir in /usr/local/cuda*_tmp; do
    mv "$cuda_dir" "${cuda_dir/_tmp/}";
  done;

  echo "=== Copying wheel for $py with CUDA ${CUDA_VERSION} to the host ===";
  readarray -t whl < <(find /tmp/src/dist -name "*.whl");
  whl_name="$(basename "$whl")";
  whl_name="${whl_name/-linux/-manylinux1}";
  mv "$whl" "${ODIR}/${whl_name}";
  wheels+=("${ODIR}/${whl_name}");
done;

echo "================================================================";
printf "=== %-56s ===\n" "Copied ${#wheels[@]} wheels to ${ODIR:5}";
echo "================================================================";
