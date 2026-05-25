#!/bin/bash
set -e

IGC_VERSION="2.24.8"
IGC_BUILD="20344"
IGC_TAG="v${IGC_VERSION}"
IGC_BASE_URL="https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}"

IGC_CORE_DEB="intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb"
IGC_OPENCL_DEB="intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb"


# Check if the current version is already installed
INSTALLED_VERSION=$(dpkg-query -W -f='${Version}' intel-igc-core-2 2>/dev/null || true)
if [[ "$INSTALLED_VERSION" == ${IGC_VERSION}* ]]; then
  echo "IGC ${IGC_TAG} is already installed (intel-igc-core-2 ${INSTALLED_VERSION}), skipping."
  exit 0
fi

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Downloading IGC ${IGC_TAG}..."
wget -q --show-progress -P "$TMPDIR" "${IGC_BASE_URL}/${IGC_CORE_DEB}"
wget -q --show-progress -P "$TMPDIR" "${IGC_BASE_URL}/${IGC_OPENCL_DEB}"

echo "Installing IGC packages..."
if [ "$(id -u)" -ne 0 ]; then
  sudo dpkg -i "$TMPDIR/${IGC_CORE_DEB}" "$TMPDIR/${IGC_OPENCL_DEB}"
else
  dpkg -i "$TMPDIR/${IGC_CORE_DEB}" "$TMPDIR/${IGC_OPENCL_DEB}"
fi

# Ensure /usr/local/lib is in ldconfig search path so 2.34.4 takes priority
if ! grep -qr '/usr/local/lib' /etc/ld.so.conf /etc/ld.so.conf.d/ 2>/dev/null; then
  echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf
fi
ldconfig
echo "IGC updated to ${IGC_TAG} successfully."
echo "Active libigc.so.2: $(ldconfig -p | grep 'libigc.so.2 ' | awk '{print $NF}')"

