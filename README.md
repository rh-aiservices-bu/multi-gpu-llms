# Distributed Serving with Multi-GPU LLMs in Kubernetes / OpenShift

This repository provides instructions for deploying LLMs with Multi-GPUs in distributed OpenShift / Kubernetes nodes.

## Overview

Serving large language models (LLMs) with multiple GPUs in a distributed environment is a challenging task. This repository provides a reference implementation for deploying LLMs with multiple GPUs in Kubernetes / OpenShift.

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