# Quantization

Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).

Reducing the number of bits means the resulting model requires less memory storage, consumes less energy (in theory), and operations like matrix multiplication can be performed much faster with integer arithmetic. It also allows to run models on embedded devices, which sometimes only support integer data types.

## Quantization Techniques

The basic idea behind quantization is quite easy: going from high-precision representation (usually the regular 32-bit floating-point) for weights and activations to a lower precision data type. The most common lower precision data types are:

* float16, accumulation data type float16
* bfloat16, accumulation data type float32
* int16, accumulation data type int32
* int8, accumulation data type int32

The two most common quantization cases are:
* [float32 -> float16](https://huggingface.co/docs/optimum/concept_guides/quantization#quantization-to-float16)
* [float16 -> int8](https://huggingface.co/docs/optimum/concept_guides/quantization#quantization-to-int8)

