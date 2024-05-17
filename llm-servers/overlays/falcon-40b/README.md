## Falcon 40B

This repository provides instructions for deploying [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b-instruct) on 8xA10G GPUs in Kubernetes / OpenShift.

### Falcon 40B Requirements

> Running the 40B model is challenging because of its size: it doesn't fit in a single A100 with 80 GB of RAM. Loading in 8-bit mode, it is possible to run in about 45 GB of RAM, which fits in an A6000 (48 GB) but not in the 40 GB version of the A100

* [Blog - Falcon 40B Inference](https://huggingface.co/blog/falcon#inference-of-falcon-40b)
* [Issue with AWS instances type g5.48xlarge](https://github.com/NVIDIA/gpu-operator/issues/634)
  * [Doc - Custom Driver Params](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/custom-driver-params.html)
  * [GSP Firmware](https://download.nvidia.com/XFree86/Linux-x86_64/510.47.03/README/gsp.html)