# EvoLLM

EvoLLM 是一个基于 PyTorch 从零实现的大语言模型学习与实验项目。

项目目标是从最基础的 Decoder-only Transformer 开始，逐步实现现代大语言模型中的核心结构，并进一步搭建完整的数据处理、预训练、监督微调、偏好对齐以及强化学习训练流程。

EvoLLM 更关注模型原理、代码实现和训练流程本身，主要用于学习、实验和研究，而不是面向生产环境部署。

## 当前进度

目前已经完成最基础的 Decoder-only Transformer，实现了以下核心模块：

- Embedding
- Multi-Head Self-Attention
- Causal Attention Mask
- LayerNorm
- Feed Forward Network
- Transformer Block

当前版本主要用于建立最基础的 Transformer 训练与推理框架，后续会逐步替换和扩展为现代 LLM 常用结构。

## 项目结构

```text
EvoLLM/
├── configs/                # 模型配置与训练配置
├── data/                   # 数据集处理
├── models/
│   ├── modules/            # Attention、RoPE、Norm 等基础模块
│   ├── evollm.py           # EvoLLM 基础模型
│   ├── evollm_tiny.py      # 小规模实验模型
│   ├── evollm_llama.py     # LLaMA 架构
│   └── evollm_qwen.py      # Qwen 架构
├── optimizer/              # 优化器与学习率调度器
├── trainer/                # Pretrain、SFT、RL 等训练流程
└── utils/                  # 日志、Checkpoint 等通用工具