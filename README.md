# AscendC 算子自动优化 Agent

基于 Claude Code 的 AscendC 昇腾算子性能自动优化框架。输入算子工程，自动完成环境检查、Shape 设计、瓶颈分析、代码改写、编译验证的闭环优化流程。

## 项目结构

```
ascendc-optim-agent/
├── .claude/                        # Agent 配置与知识库
│   ├── CLAUDE.md                   # 编排者：全局流程、规则、编码红线
│   ├── settings.json               # 模型、权限、Agent 定义
│   ├── agents/                     # 四个子 Agent
│   │   ├── shaper.md               #   Shape 设计 + 基线采集
│   │   ├── planner.md              #   瓶颈分析 + 策略输出
│   │   ├── coder.md                #   代码改写
│   │   └── builder.md              #   编译验证 + 性能采集 + 错误诊断
│   └── skills/                     # 原子知识模块（按需加载）
│       ├── env-check/              #   环境检查门禁
│       ├── shape-design/           #   Shape 设计方法
│       ├── npu-architecture/       #   NPU 架构参考
│       ├── perf-analysis/          #   性能瓶颈诊断
│       ├── kernel-patterns/        #   8 种优化代码模式
│       ├── compile-and-profile/    #   编译 + msprof 采集 + 归档
│       └── error-diagnosis/        #   编译/运行/精度三类诊断
├── log/                            # 优化日志（与源码隔离）
│   └── {OpName}/
│       ├── round_000_baseline/     #   基线：shapes.md + perf_report.md
│       ├── round_001/              #   第 1 轮：strategy.md + code_snapshot/ + perf_report.md
│       ├── round_002/
│       └── summary.md              #   全局优化摘要
├── Erfc/                           # 算子工程示例
│   ├── CMakeLists.txt
│   ├── run.sh                      # bash run.sh (全部) / bash run.sh <index> (单个)
│   ├── op_host/Erfc.asc            # Host 端（tiling 计算、ACL 调用）
│   ├── op_kernel/
│   │   ├── Erfc_kernel.asc         # Kernel 实现（流水线、Compute）
│   │   └── Erfc_tiling.h           # Tiling 常量与结构体
│   ├── scripts/                    # gen_data.py, golden.py, verify_result.py
│   └── docs/perf/round_NNN/        # msprof 性能归档（8 CSV + summary.txt）
└── README.md
```

## 优化流程

```
环境门禁 → Shaper → [Planner → Coder → Builder] 循环 → 停止
```

1. **环境门禁** — 检查 NPU 设备、CANN 环境、编译工具链、msprof
2. **Shaper** — 通读源码提取约束 → 设计边界/常见 shape → 同步更新 `run.sh` + `gen_data.py` + `op_host` → 编译验证 → 基线性能采集
3. **Planner** — 跨 shape 性能对比 → CSV 深读 → 决策树判定瓶颈 → 输出优先排序的优化策略
4. **Coder** — 按策略改写 `op_kernel/` 代码 → 备份原文件 → 自检编码红线
5. **Builder** — 编译 + 精度验证 → msprof 逐 shape 采集 → 对比 baseline → 失败时诊断修复

停止条件：高优策略耗尽 / 连续两轮改善 < 5% / 总轮数达到 3 轮。

## 日志归档

每轮优化的完整上下文自动归档到 `log/{OpName}/round_NNN/`：

| 阶段 | 归档文件 | 内容 |
|------|---------|------|
| Shaper 基线 | `round_000_baseline/shapes.md` | Shape 设计方案、维度约束、分类目的 |
| Shaper 基线 | `round_000_baseline/perf_report.md` | 基线性能汇总表 |
| Planner 策略 | `round_NNN/strategy.md` | 瓶颈判定 + 优化方案 + 预期收益 + 风险 |
| Coder 代码 | `round_NNN/code_snapshot/*.asc, *.h` | 改写后的代码快照 |
| Builder 性能 | `round_NNN/perf_report.md` | 编译状态、精度、逐 shape 对比 baseline |
| 全局 | `summary.md` | 各轮 baseline → final 关键指标对比 |

## 算子工程约定

每个算子工程遵循统一结构（以 Erfc 为例）：

```
Erfc/
├── run.sh                       # 驱动编译、数据生成、执行、验证
├── op_host/Erfc.asc             # Host 端：tiling 计算、ACL 内存管理
├── op_kernel/Erfc_kernel.asc    # Kernel：Init/Process/Compute/CopyIn/CopyOut
├── op_kernel/Erfc_tiling.h      # TilingData 结构体 + UB/BLOCK_ALIGN 常量
├── scripts/gen_data.py          # 生成输入数据和 golden
├── scripts/verify_result.py     # 精度验证（MERE/MARE 阈值）
└── docs/perf/round_NNN/         # msprof 采集结果（每 shape 一个 round）
```

**Shape 定义分散在三处**，Shaper 设计时必须同步更新：

| 文件 | 变量 | 用途 |
|------|------|------|
| `run.sh` | CASE_NAMES / CASE_DIMS / CASE_LABELS / CASE_ARGS | 驱动编译和运行 |
| `scripts/gen_data.py` | TEST_CASES / test_cases | 生成输入和 golden |
| `op_host/{op}.asc` | shape 查找表（如 shapeDim0[]） | 二进制按索引查找 shape |

## Agent 知识体系

| Skill | 提供什么 | 使用者 |
|-------|---------|--------|
| env-check | NPU/CANN/cmake/msprof 检查流程 | 编排者 |
| shape-design | 三类 Shape 设计方法 + 三文件同步更新 | Shaper |
| npu-architecture | 芯片代际、UB 容量、峰值算力、条件编译宏 | 所有 Agent |
| perf-analysis | CSV 指标解读、决策树、阈值速查、瓶颈案例 | Planner |
| kernel-patterns | 8 种优化代码模式 + API 约束 + Tiling 公式 | Coder |
| compile-and-profile | 编译/msprof 采集/perf_summary.py 归档流程 | Shaper / Builder |
| error-diagnosis | 编译/运行/精度三类诊断树 + 错误码速查 | Builder |

## 使用方式

```bash
# 1. 确保 NPU 环境就绪
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
source ${ASCEND_HOME_PATH}/set_env.sh

# 2. 在项目根目录启动 Claude Code
cd ascendc-optim-agent
claude

# 3. 指定算子工程启动优化
> 对 Erfc 算子执行优化流程
```

## 编码红线（Device 侧 op_kernel/）

- 禁止 `GlobalTensor::SetValue` / `GlobalTensor::GetValue`，用 `DataCopyPad` 批量替代
- 禁止硬编码 `blockDim` / UB 大小 / `blockIdx`，用 `GetBlockIdx()` / TilingData 传递
- 用 `DataCopyPad` 替代 `DataCopy`（除非严格 32B 对齐）
- 禁止使用 `std::` 命名空间（Device 侧无 C++ 标准库；Host 端 `op_host/` 不受此限制）
