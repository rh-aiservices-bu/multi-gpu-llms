# Distributed Serving with Multi-GPU LLMs in Kubernetes / OpenShift

This repository provides instructions for deploying LLMs with Multi-GPUs in distributed OpenShift / Kubernetes nodes.

## 1. Overview

Large LLMs like [Llama-3-70b](meta-llama/Meta-Llama-3-70B) or [Falcon 180B](https://huggingface.co/blog/falcon-180b) may not fit in a single GPU.

If training/serving a model on a single GPU is too slow or if the model’s weights do not fit in a single GPU’s memory, transitioning to a multi-GPU setup may be a viable option.

But serving large language models (LLMs) with multiple GPUs in a distributed environment might be a challenging task.

## 2. Checking the Memory Footprint of the Model

Before deploying a model in a distributed environment, it is important to check the memory footprint of the model.

To begin estimating how much vRAM is required to serve your LLM, we can use these tools:

* [HF Model Memory Usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
* [GPU Poor vRAM Calculator](https://rahulschand.github.io/gpu_poor/)
* [LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) (only for quantization models)
* [LLM Explorer](https://llm.extractum.io/) to check raw model vRAM size consumption

## 3. Using Multiple GPUs for serving an LLM

When a model is too big to fit on a single GPU, we can use [various techniques](https://huggingface.co/docs/transformers/perf_train_gpu_many#scalability-strategy) to optimize the memory utilization.

Among the different strategies, we can use [Tensor Parallelism](https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism) to distribute the model across multiple GPUs.

### 3.1 Tensor Parallelism with Serving Runtimes

Tensor parallelism is a technique used to fit large models across multiple GPUs.
In Tensor Parallelism, each GPU processes a slice of a tensor and only aggregates the full tensor for operations requiring it.

For example, when multiplying the input tensors with the first weight tensor, the matrix multiplication can be achieved by splitting the weight tensor column-wise, multiplying each column with the input separately, and then concatenating the separate outputs.

These outputs are then transferred from the GPUs and concatenated to obtain the final result.

![Tensor Parallelism](./docs/tp-diagram.png)

* [vLLM supports distributed tensor-parallel](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) inference and serving. Currently, vLLM support Megatron-LM’s tensor parallel algorithm. vLLM manage the distributed runtime with Ray. 

* [HF-TGI supports distributed tensor-parallel](https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism)

* [ODH TGI](https://github.com/opendatahub-io/text-generation-inference?tab=readme-ov-file#running-sharded-models-tensor-parallel) - Fork from IBM of [HF-TGI](https://huggingface.co/docs/text-generation-inference)

> **IMPORTANT**: Check with the AI BU PMs or your account team to ensure that the Serving Runtime you are using **supports** tensor parallelism.

There are two ways to use Tensor Parallelism:

* In a single worker node with Multiple GPUs
* Across multiple worker nodes with different GPUs allocated to each node.

## 4. vLLM Tensor Parallelism (TP)

### 4.1 vLLM TP in single Worker Node with Multiple GPUs

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

### 4.2 vLLM TP in multiple Worker Nodes with Multiple GPUs

> WIP

To scale vLLM beyond a single Worker Node, start a [Ray runtime](https://docs.ray.io/en/latest/ray-core/starting-ray.html) via CLI before running vLLM.

After that, you can run inference and serving on multiple machines by launching the vLLM process on the head node by setting tensor_parallel_size to the number of GPUs to be the total number of GPUs across all machines.

## 5. Optimizing Memory Utilization on a Single GPU

We can use [Quantization techniques](./docs/quant.md) to reduce the memory footprint of the model and try to fit the LLM in one single GPU. But there are several other techniques (like [FlashAttention-2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)) that can be used to reduce the memory footprint of the model.

> The quantization will reduce the precision and the accuracy of the model and will add some overhead to the inference time. So it's important to be aware of the that before applying quantization.

Once you have employed these strategies and found them insufficient for your case on a single GPU, consider moving to multiple GPUs.

## 6. Demos

* Running Granite 7B on 2xT4 GPUs in K8s/OpenShift
* Running Mistral 7B on 2xT4 GPUs in K8s/OpenShift
* Running Llama3 7B on 2xT4 GPUs in K8s/OpenShift
* Running Falcon 40B on 8xA10G GPUs in K8s/OpenShift

## 7. Demo Steps

### 7.1 Provision the GPU nodes in OpenShift (optional)

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

### 7.2 Deploy the Demo Use Cases

## Links of Interest

* [Distributed Serving Documentation](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
* [vLLM Engine Args](https://docs.vllm.ai/en/latest/models/engine_args.html)
* [Tensor Parallelism HF Transformers](https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism)
* [Blog - Big Inference with vLLM](https://hamel.dev/notes/llm/inference/big_inference.html)
* [VRAM Calculator](https://rahulschand.github.io/gpu_poor/)
* [vLLM with Paged Attention](https://blog.vllm.ai/2023/06/20/vllm.html)
* [Blog - AWS GPU for Deep Learning](https://towardsdatascience.com/choosing-the-right-gpu-for-deep-learning-on-aws-d69c157d8c86)