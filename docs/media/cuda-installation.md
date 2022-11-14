#### CUDA
You can download CUDA10.2 from [here](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin).  
You can follow the instruction.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

#### cuDNN
You can download specific version from [here](https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/).  
For example, the tested version is with `libcudnn8_8.0.0.180-1+cuda10.2_amd64.deb`.

Then install them using the command below.
```bash
sudo dpkg -i libcudnn8_8.0.0.180-1+cuda10.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.0.180-1+cuda10.2_amd64.deb
```
