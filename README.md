# Distributed Serving with Multi-GPU LLMs in Kubernetes / OpenShift

This repository provides instructions for deploying LLMs with Multi-GPUs in distributed OpenShift / Kubernetes nodes.

## Overview

Large LLMs like [Llama-3-70b](meta-llama/Meta-Llama-3-70B) or [Falcon 180B]() may not fit in a single GPU.

If training/serving a model on a single GPU is too slow or if the model’s weights do not fit in a single GPU’s memory, transitioning to a multi-GPU setup may be a viable option.

But serving large language models (LLMs) with multiple GPUs in a distributed environment might be a challenging task.

## Checking the Memory Footprint of the Model

Before deploying a model in a distributed environment, it is important to check the memory footprint of the model.

To begin estimating how much vRAM is required to serve your LLM, we can use three tools:

* [HF Model Memory Usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
* [GPU Poor vRAM Calculator](https://rahulschand.github.io/gpu_poor/)
* [LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) (only for quantization models)
* [LLM Explorer](https://llm.extractum.io/) to check raw model vRAM size consumption

## Optimizing Memory Utilization on a Single GPU

When a model is too big to fit on a single GPU, we can use [various techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one) to optimize the memory utilization.

We can use [Quantization techniques](./docs/quant.md) to reduce the memory footprint of the model. But there are several other techniques (like [FlashAttention-2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)) that can be used to reduce the memory footprint of the model.

> The quantization will reduce the precision and the accuracy of the model and will add some overhead to the inference time. So it's important to be aware of the that before applying quantization.

Once you have employed these strategies and found them insufficient for your case on a single GPU, consider moving to multiple GPUs.

## Demos

* Running Mistral 7B on 2xT4 GPUs in K8s/OpenShift
* Running Llama3 7B on 2xT4 GPUs in K8s/OpenShift
* Running Falcon 40B on 8xA10G GPUs in K8s/OpenShift

## Usage

TBD

## Links of Interest

* [Distributed Serving Documentation](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
* [vLLM Engine Args](https://docs.vllm.ai/en/latest/models/engine_args.html)
* [Tensor Parallelism HF Transformers](https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism)
* [Blog - Big Inference with vLLM](https://hamel.dev/notes/llm/inference/big_inference.html)
* [VRAM Calculator](https://rahulschand.github.io/gpu_poor/)
* [vLLM with Paged Attention](https://blog.vllm.ai/2023/06/20/vllm.html)