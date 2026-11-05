# ChineseNameGenerator

用字符级上下文 + 多层感知机（MLP）生成中文姓名。项目改写自 Andrej Karpathy 的
`makemore` 系列，把英文的字符级 bigram / MLP 语言模型换成中文汉字，并扩展为
可配置上下文长度、网络深度和学习率调度的版本。

## 项目简介

模型的任务是：给定前 `LenContext` 个汉字，预测下一个汉字。

每个名字被改写成：

```
. 张 伟 .
```

其中 `.` 是边界符（索引固定为 0），既表示姓名开始，也表示姓名结束。模型从全 `.`
的上下文出发，逐字采样，采到 `.` 或达到最大长度就停止。

核心流程：

1. **读取语料**：`ChineseName1.txt`，一行一个姓名，UTF-8 编码。
2. **建立映射**：所有出现过的汉字按 Unicode 排序编号为 1…N，`.` 固定为 0。
3. **构造样本**：滑动上下文窗口，得到 `X: (样本数, LenContext)` 和 `Y: (样本数,)`。
4. **前向计算**：查嵌入表 → 展平 → 若干隐藏层（`tanh`）→ 输出层（`vocab_size` 个 logits）。
5. **训练**：交叉熵损失 + 反向传播 + 手写 SGD，学习率按所选策略衰减。
6. **采样**：对 logits 做 softmax，按概率抽取下一个字，并把上下文右移一格。

当前配置下的规模：

| 项目 | 数值 |
|---|---|
| 姓名数 | 3,548,868 |
| 不同汉字 | 6,505 |
| `vocab_size`（含 `.`） | 6,506 |
| 上下文长度 `LenContext` | 2 |
| 嵌入维度 `DimEmb` | 10 |
| 网络结构 | 20 → 200 → 200 → 6506 |
| 训练样本数 | 约 1,390 万 |
| 训练步数 | 20,000 |
| 训练结束时的 batch loss | 约 4.5–5.2 |

初始 loss 约 8.8，接近均匀分布基线 `ln(6506) ≈ 8.78`。

## 目录结构

```
Week2/
├── Note/
│   └── Week2-C-notes.pdf        # 第 2 周课程笔记（MLP、损失、SGD、数据划分）
└── ChineseNameGenerator/
    ├── CG.ipynb                 # 主程序
    ├── ChineseName1.txt         # 姓名语料，一行一个
    └── README.md
```

## 环境要求

- Python 3.12（开发时使用 3.12.13）
- PyTorch 2.15.0
- Jupyter Notebook / JupyterLab
- 不需要 GPU；当前模型规模用 CPU 即可训练

启动后**按顺序运行所有单元格**。数据文件使用相对路径 `ChineseName1.txt`，
因此 notebook 的工作目录必须是 `Week2/ChineseNameGenerator`。

## 数据集

`ChineseName1.txt` 来自 CCNC（Comprehensive Chinese Name Corpus）：

- 仓库：https://github.com/jaaack-wang/ccnc
- 原始规模：3,658,109 条，制表符分隔的四列 `姓 / 名 / 全名 / 性别`
- 清洗方式：取“全名”列，用 `[一-鿿]{2,3}` 只保留 2–3 个纯汉字，去重
- 清洗结果：3,548,868 条（2 字名 299,836，3 字名 3,249,032）
- 许可：GPL-3.0

notebook 中还会再做一次正则过滤，并用固定种子打乱顺序。

## 可调参数

所有参数集中在 **参数单元**（`DimEmb` 所在的那个代码单元）。修改后需要从该单元
开始重新运行，训练单元会重新构造样本。

### 模型结构

| 参数 | 当前值 | 说明 |
|---|---|---|
| `DimEmb` | `10` | 每个汉字的嵌入维度。增大可提高表示能力，同时增加参数量和计算量，常用 16–64 |
| `LenContext` | `2` | 用前多少个字预测下一个字。中文姓名通常 2–3 字，取 1–3 较合理 |
| `layer_dims` | `[20, 200, 200, 6506]` | 每一层的维度。第一个数必须等于 `DimEmb * LenContext`，最后一个必须等于 `vocab_size`；中间可自由增删 |
| `g` | `manual_seed(2147483647)` | 随机种子，控制初始化、batch 采样和生成结果的可复现性 |

`layer_dims` 的长度减 1 就是权重层数：

```python
layer_dims = [DimEmb * LenContext, 200, 200, vocab_size]
# 3 层：20 → 200 → 200 → 6506
```

- `[20, 2271]`：无隐藏层，等价于多分类线性模型
- `[20, 200, 2271]`：1 个隐藏层
- `[20, 200, 200, 2271]`：2 个隐藏层
- 隐藏层变宽或变深都会增加容量，但也更容易过拟合、训练更慢

注意：修改 `DimEmb` 或 `LenContext` 后，`layer_dims` 的第一个数要同步修改，
notebook 里的 `assert` 会检查这一点。

### 初始化

隐藏层权重乘以 `0.2`，输出层乘以 `0.01`，偏置初始化为 0。

深层网络如果直接用标准正态初始化，前向激活和梯度容易逐层放大；缩放初始化能让
初始 loss 接近 `ln(vocab_size)`，训练更稳定。层数越多，这个缩放越重要。

### 训练

| 参数 | 当前值 | 说明 |
|---|---|---|
| `max_steps` | `20000` | SGD 更新次数。每个 step 只取一个 minibatch，不是完整 epoch |
| `lr_init` | `1` | 初始学习率。当前梯度是 batch 平均，学习率可以比小模型大 |
| `lr_schedule` | `'exponential'` | 学习率衰减方式，见下表 |
| batch size | `32`（硬编码） | 在训练单元的 `torch.randint(0, X.shape[0], (32,))` 中修改；常用 32–256 |

`lr_schedule` 的可选值：

| 取值 | 公式 | 行为 |
|---|---|---|
| `'constant'` | `lr = lr_init` | 全程不变 |
| `'step'` | `lr_init * 0.01 ** (i // (max_steps // 5))` | 每 1/5 训练过程降为原来的 1/100 |
| `'exponential'` | `lr_init * 0.01 ** (i / max_steps)` | 连续指数衰减，最终为初始值的 1/100 |
| `'cosine'` | `0.5 * lr_init * (1 + cos(pi * i / max_steps))` | 余弦退火，最终到 0 |
| `'linear'` | `lr_init * (1 - i / max_steps)` | 线性衰减，最终到 0 |

如果学习率太大，loss 会震荡甚至发散；太小则 20,000 步内学不完。可以先固定
`lr_schedule='constant'` 试几个 `lr_init`，再换成衰减策略。

### 生成

| 参数 | 当前值 | 说明 |
|---|---|---|
| `generate(max_len=3)` | `3` | 最多生成几个字（不含结尾的 `.`） |
| 前两字禁止结束 | 固定 | 保证生成结果至少 2 个字 |

采样使用 `torch.multinomial`，结果是随机的；固定 `g` 的种子可以复现。训练过程中
`g` 也被用来抽 batch，所以改变训练步数会间接改变生成结果。

### 数据

| 参数 | 当前值 | 说明 |
|---|---|---|
| 文件名 | `ChineseName1.txt` | 换成语料时修改读取单元中的路径 |
| 过滤规则 | `[一-鿿]{2,3}` | 只保留 2–3 个纯汉字；改成长度范围时要同步调整 `generate(max_len=...)` |

## 运行说明与常见问题

- **内存**：3.55M 个名字大约产生 1,390 万个样本。notebook 用 Python list 拼
  `X_flat` / `Y`，峰值内存约 1.5–2 GB。内存不足时，可以先对 `words` 随机抽样
  （例如 `words = words[:500000]`），或改成分批构造样本。
- **输出很长**：训练单元每一步都会 `print(i, ' : ', loss.item())`，20,000 步会
  产生两万行输出。不需要逐步观察时，可以注释掉这一行。
- **旧 bigram 单元**：构造 `xs` / `ys` 的那个单元是 part1 的遗留代码，MLP 训练
  并不使用它，可以跳过或删除；它反而会额外占用数百 MB 内存。
- **修改结构后**：从参数单元开始重新运行；如果换了数据集，需要从头运行，因为
  `stoi` / `itos` / `vocab_size` 都会变。
- **loss 多少算正常**：初始应接近 `ln(vocab_size)`；训练结束时当前配置约为
  4.5–5.2。显著高于这个范围通常说明学习率、初始化或数据出了问题。

## 参考

课程与讲义：

- Andrej Karpathy, *Neural Networks: Zero to Hero*, Lecture 2: Bigrams
  - 视频：https://www.youtube.com/watch?v=PaCmpygFfXo
  - notebook：`lectures/makemore/makemore_part1_bigrams.ipynb`
- Andrej Karpathy, *Neural Networks: Zero to Hero*, Lecture 3: MLP
  - 视频：https://youtu.be/TCH_1BHY58I
  - notebook：`lectures/makemore/makemore_part2_mlp.ipynb`
- Aaron Mazel-Gee, *Neural Networks: Zero to Hero: A Mathematician's Companion*,
  2024-04-09, §3.1–3.7, pp. 14–19
  - 本地笔记：`Week2/Note/Week2-C-notes.pdf`

数据：

- CCNC: A Comprehensive Chinese Name Corpus
  - https://github.com/jaaack-wang/ccnc
  - 许可：GPL-3.0
- wainshine/Chinese-Names-Corpus（另一份中文姓名语料，Apache-2.0）
  - https://github.com/wainshine/Chinese-Names-Corpus
- psychbruce/ChineseNames（1930–2008 全国姓名统计，可用于校验用字频率）
  - https://github.com/psychbruce/ChineseNames

工具文档：

- PyTorch `torch.multinomial`：
  https://pytorch.org/docs/stable/generated/torch.multinomial.html
- PyTorch `torch.nn.functional.cross_entropy`：
  https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- PyTorch 自动求导：
  https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
