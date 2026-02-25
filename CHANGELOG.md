# Changelog

All notable changes to GPULlama3.java will be documented in this file.

## [0.4.0] - 2026-02-25

### Other Changes

-   Add JDK 25 support with TornadoVM JDK25 and dual-JDK build profiles ([#97](https://github.com/beehive-lab/GPULlama3.java/pull/97))

## [0.3.3] - 2025-12-19

<!-- TODO: Add changes manually -->

## [0.3.2] - 2025-12-18

### Model Support

- [models] Support for IBM Granite Models 3.2, 3.3 & 4.0 with FP16 and Q8 ([#92](https://github.com/beehive-lab/GPULlama3.java/pull/92))

### Other Changes

- [docs] Update docs to use SDKMAN! and point to TornadoVM 2.2.0 ([#93](https://github.com/beehive-lab/GPULlama3.java/pull/93))
- Add JBang catalog and local usage examples to README.md ([#91](https://github.com/beehive-lab/GPULlama3.java/pull/91))
- Add `jbang` script and configuration to make easy to run ([#90](https://github.com/beehive-lab/GPULlama3.java/pull/90))

## [0.3.1] - 2025-12-11

### Model Support

- Add compatibility method for langchain4j and quarkus in ModelLoader ([#87](https://github.com/beehive-lab/GPULlama3.java/pull/87))

## [0.3.0] - 2025-12-11

### Model Support

- [refactor] Generalize the design of `tornadovm` package to support multiple new models and types for GPU exec  ([#62](https://github.com/beehive-lab/GPULlama3.java/pull/62))
- Refactor/cleanup model loaders ([#58](https://github.com/beehive-lab/GPULlama3.java/pull/58))
- Add Support for Q8_0 Models ([#59](https://github.com/beehive-lab/GPULlama3.java/pull/59))

### Bug Fixes

- [fix] Normalization compute step for non-nvidia hardware ([#84](https://github.com/beehive-lab/GPULlama3.java/pull/84))

### Other Changes

- Update README to enhance TornadoVM performance section and clarify GP… ([#85](https://github.com/beehive-lab/GPULlama3.java/pull/85))
- Simplify installation by replacing TornadoVM submodule with pre-built SDK ([#82](https://github.com/beehive-lab/GPULlama3.java/pull/82))
- [FP16] Improved performance by fusing dequantize with compute  in kernels: 20-30% Inference Speedup ([#78](https://github.com/beehive-lab/GPULlama3.java/pull/78))
- [cicd] Prevent workflows from running on forks ([#83](https://github.com/beehive-lab/GPULlama3.java/pull/83))
- [CI][packaging] Automate process of deploying a new release with Github actions ([#81](https://github.com/beehive-lab/GPULlama3.java/pull/81))
- [Opt] Manipulation of Q8_0 tensors with Tornado `ByteArray`s ([#79](https://github.com/beehive-lab/GPULlama3.java/pull/79))
- Optimization in Q8_0 loading ([#74](https://github.com/beehive-lab/GPULlama3.java/pull/74))
- [opt] GGUF Load Optimization for tensors in TornadoVM layout ([#71](https://github.com/beehive-lab/GPULlama3.java/pull/71))
- Add `SchedulerType` support to all TornadoVM layer planners and layer… ([#66](https://github.com/beehive-lab/GPULlama3.java/pull/66))
- Weight Abstractions ([#65](https://github.com/beehive-lab/GPULlama3.java/pull/65))
- Bug fixes in sizes and names of GridScheduler ([#64](https://github.com/beehive-lab/GPULlama3.java/pull/64))
- Add Maven wrapper support ([#56](https://github.com/beehive-lab/GPULlama3.java/pull/56))
- Add changes used in Devoxx Demo ([#54](https://github.com/beehive-lab/GPULlama3.java/pull/54))

