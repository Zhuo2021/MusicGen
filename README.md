# MusicGen Agent

这是一个面向 MIDI 旋律续写的音乐生成 Agent。它会从本地 MIDI 语料中抽取旋律短语，基于音乐特征做检索增强，然后生成一段风格连续的 MIDI 续写，并输出可量化的音乐性评估指标。


## 亮点

- **RAG for Music**：把 MIDI 切成 phrase，用音程直方图、pitch class、旋律轮廓、音域等特征做相似片段检索。
- **离线可运行**：默认不依赖 OpenAI key，不会因为外部 API 不可用导致 demo 崩掉。
- **Agent 工作流清晰**：`seed melody -> corpus indexing -> retrieval -> continuation planning -> MIDI rendering -> evaluation`。
- **工程化结构**：核心逻辑拆到 `src/musicgen_agent`，保留 `MusicGen.py` 作为兼容入口。
- **可评估**：生成后输出音符数、音域、平均音程、密度、调式、重复率、大跳率等指标，方便和 RNN/LSTM 结果横向比较。
- **面试可讲**：既能讲音乐理论特征，也能讲检索增强、模块设计、fallback 策略和测试。

## 快速运行

```bash
python MusicGen.py --seed G4,E-4,D4,G3 --target-notes 24 --output outputs/generated.mid
```

也可以安装成命令行工具：

```bash
pip install -e .
musicgen-agent --seed C4,E4,G4,C5 --emotion q1 --output outputs/demo.mid
```

运行后会看到：

- 生成的完整音符序列
- Top retrieval hits 及相似度解释
- 评估指标
- 写出的 MIDI 文件路径

## 项目结构

```text
MusicGen.py                    # 兼容入口，一条命令跑 demo
src/musicgen_agent/
  agent.py                     # Agent 编排和续写策略
  corpus.py                    # MIDI 语料加载与 phrase 切分
  midi_io.py                   # 轻量 MIDI 读写，不强依赖第三方库
  retrieval.py                 # 音乐特征检索
  theory.py                    # 音名解析、调性/轮廓/音程特征
  evaluation.py                # 生成结果评估
  cli.py                       # 命令行入口
tests/                         # 回归测试
lstm生成音乐/                  # 原始 LSTM 训练资料和 MIDI 数据
Music_Gen_v1/                  # 早期 RAG 原型和 ChromaDB 资产
```

## 设计取舍

早期版本把 RAG 演示、ChromaDB、Prompt 和 OpenAI 调用写在一个脚本里，适合说明想法，但不适合给面试官直接运行。现在的版本把 LLM 依赖降级为可选项，先用确定性的 symbolic composer 保证 demo 稳定，再保留后续接入 LLM planner / MusicBERT embedding / ChromaDB 的空间。

当前实现重点关注旋律续写。节奏使用检索片段的 rhythm pattern 迁移，不再全部等长；音高生成会参考相似 phrase 的 interval motion，并做可演奏音域约束，避免生成结果突然跳到不自然的极端音区。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

测试覆盖了 MIDI 读写回环、语料切分和 Agent 生成长度，保证后续继续优化模型时不会把基础数据流改坏。

## 后续计划

- 接入 EMOPIA 四象限情绪标签，做 emotion-aware retrieval。
- 增加 LSTM / RNN baseline 的统一评估脚本，输出对比表。
- 支持多轨 MIDI 和 chord-aware continuation。
- 将当前 deterministic planner 扩展为可选 LLM planner，但保留离线 fallback。
