## Testing Multi-GPU Demos

## Check the GPU nodes consumption

```bash
POD_NAME=$(kubectl get pod -n nvidia-gpu-operator -l app=nvidia-device-plugin-daemonset -o jsonpath="{.items[0].metadata.name}")
kubectl exec -n nvidia-gpu-operator $POD_NAME -- nvidia-smi
```

For example this is the output of the demo of Falcon 7B with 2xTesla T4 GPUs:

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1B.0 Off |                    0 |
| N/A   27C    P8              14W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla T4                       On  | 00000000:00:1C.0 Off |                    0 |
| N/A   31C    P0              26W /  70W |   7215MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla T4                       On  | 00000000:00:1D.0 Off |                    0 |
| N/A   31C    P0              26W /  70W |   7215MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   27C    P8              14W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

## Query the LLM served by Multi-GPU

* Check [vllm_rest_requests.ipynb](./vllm_rest_requests.ipynb) for more details on how to query the LLM model.