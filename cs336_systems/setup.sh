pip install uv
apt update -y
apt install -y
apt install -y wget gnupg

wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt install -y nsight-systems-2025.6.3