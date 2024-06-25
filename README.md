# Distributed Serving with Multi-GPU LLMs in OpenShift

This repository provides instructions for deploying LLMs with Multi-GPUs in distributed OpenShift / Kubernetes worker nodes.

## Table of Contents
- [Overview](#1-overview)
- [Important Disclaimer](#2-important-disclaimer)
- [Checking the Memory Footprint of the Model](#3-checking-the-memory-footprint-of-the-model)
- [Using Multiple GPUs for Serving an LLM](#4-using-multiple-gpus-for-serving-an-llm)
  - [Tensor Parallelism with Serving Runtimes](#41-tensor-parallelism-with-serving-runtimes)
- [vLLM Tensor Parallelism (TP)](#5-vllm-tensor-parallelism-tp)
  - [vLLM TP in Single Worker Node with Multiple GPUs](#51-vllm-tp-in-single-worker-node-with-multiple-gpus)
  - [vLLM TP in Multiple Worker Nodes with Multiple GPUs](#52-vllm-tp-in-multiple-worker-nodes-with-multiple-gpus)
- [Optimizing Memory Utilization on a Single GPU](#6-optimizing-memory-utilization-on-a-single-gpu)
- [Demos](#7-demos)
  - [Single Node - Multiple GPU Demos](#71-single-node---multiple-gpu-demos)
  - [Multi-Node - Multiple GPU Demos](#72-multi-node---multiple-gpu-demos)
- [Demo Steps](#8-demo-steps)
  - [Provision the GPU nodes in OpenShift (optional)](#81-provision-the-gpu-nodes-in-openshift-optional)
  - [Deploy the Demo Use Cases](#82-deploy-the-demo-use-cases)
    - [Deploy the Single Node - Multiple GPU Demos](#821-deploy-the-single-node---multiple-gpu-demos)
    - [Deploy the Multi-Node - Multiple GPU Demos](#822-deploy-the-multi-node---multiple-gpu-demos)
- [Testing the Multi-GPU Demos](#9-testing-the-multi-gpu-demos)
- [Links of Interest](#10-links-of-interest)

## 1. Overview

Large LLMs like [Llama-3-70b](meta-llama/Meta-Llama-3-70B) or [Falcon 180B](https://huggingface.co/blog/falcon-180b) may not fit in a single GPU.

If training/serving a model on a single GPU is too slow or if the model’s weights do not fit in a single GPU’s memory, transitioning to a multi-GPU setup may be a viable option.

But serving large language models (LLMs) with multiple GPUs in a distributed environment might be a challenging task.

## 2. Important Disclaimer

> IMPORTANT DISCLAIMER: Read before proceed!

* These demos/repository are **not supported by OpenShift AI/RHOAI**; they rely on upstream projects.
* This is prototyping/testing work intended to confirm functionality and determine the necessary requirements.
* These features are **not available in the RHOAI dashboard**. If you want to implement them, you will need to adapt YAML files to fit your use case.

## 3. Checking the Memory Footprint of the Model

Before deploying a model in a distributed environment, it is important to check the memory footprint of the model.

To begin estimating how much vRAM is required to serve your LLM, we can use these tools:

* [HF Model Memory Usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
* [GPU Poor vRAM Calculator](https://rahulschand.github.io/gpu_poor/)
* [LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) (only for quantization models)
* [LLM Explorer](https://llm.extractum.io/) to check raw model vRAM size consumption

## 4. Using Multiple GPUs for serving an LLM

When a model is too big to fit on a single GPU, we can use [various techniques](https://huggingface.co/docs/transformers/perf_train_gpu_many#scalability-strategy) to optimize the memory utilization.

Among the different strategies, we can use [Tensor Parallelism](https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism) to distribute the model across multiple GPUs.

### 4.1 Tensor Parallelism with Serving Runtimes

Tensor parallelism is a technique used to fit large models across multiple GPUs.
In Tensor Parallelism, each GPU processes a slice of a tensor and only aggregates the full tensor for operations requiring it.

For example, when multiplying the input tensors with the first weight tensor, the matrix multiplication can be achieved by splitting the weight tensor column-wise, multiplying each column with the input separately, and then concatenating the separate outputs.

These outputs are then transferred from the GPUs and concatenated to obtain the final result.

![Tensor Parallelism](./docs/tp-diagram.png)

* [vLLM supports distributed tensor-parallel](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) inference and serving. Currently, vLLM support Megatron-LM’s tensor parallel algorithm. vLLM manage the distributed runtime with Ray or Python MultiProcessing.

* [HF-TGI supports distributed tensor-parallel](https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism)

* [ODH TGI](https://github.com/opendatahub-io/text-generation-inference?tab=readme-ov-file#running-sharded-models-tensor-parallel) - Fork from IBM of [HF-TGI](https://huggingface.co/docs/text-generation-inference)

> **IMPORTANT**: Check with the AI BU PMs or your account team to ensure that the Serving Runtime you are using **supports** tensor parallelism.

There are two ways to use Tensor Parallelism:

* In a single worker node with Multiple GPUs
* Across multiple worker nodes with different GPUs allocated to each node.

## 5. vLLM Tensor Parallelism (TP)

Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.

### 5.1 vLLM TP in single Worker Node with Multiple GPUs

To run [multi-GPU serving in one single Worker Node](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) (with multiple GPUs), pass in the --tensor-parallel-size argument when starting the server. This argument specifies the number of GPUs to use for tensor parallelism.

* For example, to run Mistral 7B with 2 GPUs, use the following command:

```bash
  - "--model"
  - mistralai/Mistral-7B-Instruct-v0.2
  - "--download-dir"
  - /models-cache
  - "--dtype"
  - float16
  - "--tensor-parallel-size=2"
```

### 5.2 vLLM TP in multiple Worker Nodes with Multiple GPUs

> WIP

To [scale vLLM beyond a single Worker Node](https://docs.vllm.ai/en/stable/serving/distributed_serving.html), start a [Ray runtime](https://docs.ray.io/en/latest/ray-core/starting-ray.html) via CLI before running vLLM.

After that, you can run inference and serving on multiple machines by launching the vLLM process on the head node by setting tensor_parallel_size to the number of GPUs to be the total number of GPUs across all machines.

## 6. Optimizing Memory Utilization on a Single GPU

We can use [Quantization techniques](./docs/quant.md) to reduce the memory footprint of the model and try to fit the LLM in one single GPU. But there are several other techniques (like [FlashAttention-2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)) that can be used to reduce the memory footprint of the model.

> The quantization will reduce the precision and the accuracy of the model and will add some overhead to the inference time. So it's important to be aware of the that before applying quantization.

Once you have employed these strategies and found them insufficient for your case on a single GPU, consider moving to multiple GPUs.

## 7. Demos

## 7.1 Single Node - Multiple GPU Demos

* [Running Granite 8B on 2xT4 GPUs](./llm-servers/overlays/granite-8b/README.md)
* [Running Mistral 7B on 2xT4 GPUs](./llm-servers/overlays/mistral-7b/README.md)
* [Running Llama3 7B on 2xT4 GPUs](./llm-servers/overlays/llama3-7b/README.md)
* [Running Falcon 40B on 8xA10G GPUs](./llm-servers/overlays/falcon-40b/README.md)
* [Running Mixtral 8x7B on 8xA10G GPUs](./llm-servers/overlays/mixtral-8x7b/README.md)
* [Running Llama2 13B on 2xA10G GPUs](./llm-servers/overlays/llama2-13b/README.md)

## 7.2 Multi-Node - Multiple GPU Demos

TBD

## 8. Demo Steps

### 8.1 Provision the GPU nodes in OpenShift (optional)

> If you have already GPUs installed in your OpenShift cluster, you can skip this step.

* Provision the GPU nodes in OpenShift / Kubernetes using a MachineSet

```bash
bash bootstrap/gpu-machineset.sh
```

* Follow the instructions in the script to provision the GPU nodes.

```bash
### Select the GPU instance type:
1) Tesla T4 Single GPU  4) A10G Multi GPU       7) DL1
2) Tesla T4 Multi GPU   5) A100                 8) L4 Single GPU
3) A10G Single GPU      6) H100                 9) L4 Multi GPU
Please enter your choice: 3
### Enter the AWS region (default: us-west-2): 
### Select the availability zone (az1, az2, az3):
1) az1
2) az2
3) az3
Please enter your choice: 3
### Creating new machineset worker-gpu-g5.2xlarge-us-west-2c.
machineset.machine.openshift.io/worker-gpu-g5.2xlarge-us-west-2c created
--- New machineset worker-gpu-g5.2xlarge-us-west-2c created.
```

### 8.2 Deploy the Demo Use Cases

* Create the Namespace for the demo:

```bash
kubectl create ns multi-gpu-poc
```

#### 8.2.1 Deploy the Single Node - Multiple GPU Demos

* For example if you want to deploy the Granite 7B model on 2xT4 GPUs, run the following command:

```bash
kubectl apply -k llm-servers/overlays/granite-7B/
```

> Check the README.md file in each overlay folder for more details on how to deploy the model.

#### 8.2.2 Deploy the Multi-Node - Multiple GPU Demos

TBD

### 9. Testing the Multi-GPU Demos

* Check the [Testing Multi-GPU Demos section](./test-notebooks/README.md) for more details on how to test the deployed models.

## 10. Links of Interest

* [Distributed Serving Documentation](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
* [vLLM Engine Args](https://docs.vllm.ai/en/latest/models/engine_args.html)
* [Tensor Parallelism HF Transformers](https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism)
* [Blog - Big Inference with vLLM](https://hamel.dev/notes/llm/inference/big_inference.html)
* [VRAM Calculator](https://rahulschand.github.io/gpu_poor/)
* [vLLM with Paged Attention](https://blog.vllm.ai/2023/06/20/vllm.html)
* [Blog - AWS GPU for Deep Learning](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86)
