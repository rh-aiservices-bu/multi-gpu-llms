# Mixtral 8x7B on 4xA10G GPUs (WIP)

This demo shows how to deploy the Mixtral 8x7B model on 4xA10G GPUs.

### Usage

* Deploy the Mixtral 8x7B model on 4xA10G GPUs:

```bash
kubectl apply -k llm-servers/overlays/mixtral-8x7B
```

* Remember to add your HUGGING_FACE_HUB_TOKEN into the Environment Variables to be able to download the model from the Hugging Face Hub.

* Check that the LLM is running properly:

```bash
kubectl get pod -n multi-gpu-poc
NAME                   READY   STATUS    RESTARTS   AGE
llm1-f687846b9-68bvq   1/1     Running   0          2m1s
```

* Check the logs of the Pod LLM:

```bash
kubectl logs -n multi-gpu-poc -l app=llm1
```

* Check the NVIDIA GPU consumption:

```bash
POD_NAME=$(kubectl get pod -n nvidia-gpu-operator -l app=nvidia-device-plugin-daemonset -o jsonpath="{.items[0].metadata.name}")
kubectl exec -n nvidia-gpu-operator $POD_NAME -- nvidia-smi
```

* The output should be similar to:

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    On  |   00000000:00:16.0 Off |                    0 |
|  0%   29C    P0             70W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A10G                    On  |   00000000:00:17.0 Off |                    0 |
|  0%   28C    P0             68W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A10G                    On  |   00000000:00:18.0 Off |                    0 |
|  0%   27C    P0             68W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A10G                    On  |   00000000:00:19.0 Off |                    0 |
|  0%   28C    P0             68W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A10G                    On  |   00000000:00:1A.0 Off |                    0 |
|  0%   28C    P0             68W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA A10G                    On  |   00000000:00:1B.0 Off |                    0 |
|  0%   27C    P0             69W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA A10G                    On  |   00000000:00:1C.0 Off |                    0 |
|  0%   27C    P0             69W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A10G                    On  |   00000000:00:1D.0 Off |                    0 |
|  0%   28C    P0             69W /  300W |   10364MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

* Create a Workbench and clone the repo. Execute the [llm_rest_requests.ipynb](../../../test-notebooks/llm_rest_requests.ipynb) notebook to query the LLM model.

### Fix for AWS g5.48xlarge

```bash
kubectl create configmap kernel-module-params -n gpu-operator --from-file=nvidia.conf=./bootstrap/nvidia.conf
oc patch clusterpolicy/gpu-cluster-policy -n nvidia-gpu-operator --type='json' -p='[{"op": "add", "path": "/spec/driver/kernelModuleConfig/name", "value":"kernel-module-params"}]'
```

> This patch is needed to fix the issue with the AWS instances type g5.48xlarge. The issue is related to the kernel module parameters that need to be set for the NVIDIA driver to work properly. Don't rush, will take at least 10 minutes for the nodes to be ready to be consumed by the LLM.

* [Issue with AWS instances type g5.48xlarge](https://github.com/NVIDIA/gpu-operator/issues/634#issuecomment-1876847722)
  * [Doc - Custom Driver Params](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/custom-driver-params.html)
  * [GSP Firmware](https://download.nvidia.com/XFree86/Linux-x86_64/510.47.03/README/gsp.html)