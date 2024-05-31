## Llama3-7B on 2xT4 GPUs

This demo shows how to deploy the Mistral 7B model on 2xT4 GPUs.

### Usage

* Deploy the Llama3 7B model on 2xT4 GPUs:

```bash
kubectl apply -k llm-servers/overlays/llama3-7B
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

> The output should be similar to:

```bash
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%
```

* Check the NVIDIA GPU consumption:

```bash
POD_NAME=$(kubectl get pod -n nvidia-gpu-operator -l app=nvidia-device-plugin-daemonset -o jsonpath="{.items[0].metadata.name}")
kubectl exec -n nvidia-gpu-operator $POD_NAME -- nvidia-smi
```

* The output should be similar to:

```bash
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1B.0 Off |                    0 |
| N/A   34C    P0              27W /  70W |  12951MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla T4                       On  | 00000000:00:1C.0 Off |                    0 |
| N/A   34C    P0              26W /  70W |  12921MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla T4                       On  | 00000000:00:1D.0 Off |                    0 |
| N/A   30C    P8              14W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   30C    P8              14W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

* Create a Workbench and clone the repo. Execute the [llm_rest_requests.ipynb](../../../test-notebooks/llm_rest_requests.ipynb) notebook to query the LLM model.
