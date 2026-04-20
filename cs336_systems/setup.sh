pip install uv
apt update
apt install vim
apt install wget gnupg

wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt install nsight-systems-2025.6.3