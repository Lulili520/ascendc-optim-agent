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
│       ├── perf-analysis/          #   性能瓶颈诊断决策树
│       ├── perf-collection/        #   msprof 采集 + perf_summary.py 归档
│       ├── precision-verify/       #   精度验证标准（MERE/MARE）
│       ├── kernel-patterns/        #   8 种优化代码模式 + 编码红线
│       └── error-diagnosis/        #   编译/运行/精度诊断树
├── Cummin/                         # 算子工程（只读，不被修改）
│   ├── CMakeLists.txt
│   ├── op_host/
│   │   ├── Cummin.asc              #   Host 端（tiling 计算、ACL 调用）
│   │   └── data_utils.h
│   ├── op_kernel/
│   │   ├── Cummin_kernel.asc       #   Kernel 实现
│   │   └── Cummin_tiling.h         #   Tiling 常量与结构体
│   └── scripts/
│       ├── gen_data.py             #   生成输入数据和 golden
│       ├── golden.py               #   Golden 参考实现
│       └── verify_result.py        #   精度验证
├── log/                            # 优化日志（与源码隔离）
│   └── Cummin/
│       ├── round_000/              #   Shaper 基线（完整工程副本）
│       │   ├── op_kernel/ op_host/ scripts/
│       │   ├── build/ input/ output/
│       │   ├── msprof/shape_0..2/  #   各 shape 性能数据
│       │   ├── shapes.md           #   Shape 设计方案
│       │   └── perf_report.md      #   基线性能汇总
│       ├── round_001/              #   第 1 轮优化
│       │   ├── op_kernel/ op_host/ scripts/  #   修改后完整源码
│       │   ├── build/ input/ output/
│       │   ├── msprof/shape_*/
│       │   ├── strategy.md         #   Planner 策略
│       │   └── perf_report.md
│       ├── round_002/ .. round_005/
│       └── summary.md              #   全局优化摘要
└── README.md
```

## 优化流程

```
环境门禁 → Shaper → [Planner → Coder → Builder] 循环 → 停止
```

1. **环境门禁** (`/env-check`) — 检查 NPU 设备、CANN 环境、编译工具链、msprof。任一项失败立即终止。
2. **Shaper** — 读源码提取约束 → 设计边界/常见/压力 3 类 Shape → 复制到 `round_000` → 编译验证 → 基线性能采集
3. **Planner** — 读上一轮源码和性能 → 跨 Shape 对比 → CSV 深读 → 决策树判定瓶颈 → 输出优先排序的优化策略 (`strategy.md`)
4. **Coder** — 复制上轮代码到新 round → 按策略改写 `op_kernel/*` 和 `op_host/*` → 自检编码红线
5. **Builder** — 编译 + 精度验证 → msprof 逐 shape 采集 → 归档性能 → 对比上轮 → 失败时诊断修复

停止条件：策略耗尽 / 连续两轮改善 < 5% / 总轮数达到 3 轮。

## 四 Agent 闭环

```
/env-check 通过
      │
      ▼
┌───────────────────────┐
│  Shaper（基线阶段）     │
│  /shape-design         │
│  /precision-verify     │
│  /perf-collection      │
│                       │
│  设计 3 类 shape        │
│  → 复制到 round_000     │
│  → 编译验证+采集基线    │
└───────────┬───────────┘
            │
      ┌─────▼─────┐
      │  循环开始   │
      └─────┬─────┘
            │
┌───────────▼───────────┐
│  Planner（瓶颈分析）    │
│  /perf-analysis         │
│  → strategy.md          │
└───────────┬───────────┘
            │
┌───────────▼───────────┐
│  Coder（代码改写）      │
│  /kernel-patterns       │
│  → op_kernel/ + op_host/│
└───────────┬───────────┘
            │
┌───────────▼───────────┐
│  Builder（验证+采集）   │
│  /precision-verify      │
│  /perf-collection       │
│  /error-diagnosis       │
│  → perf_report.md       │
└───────────┬───────────┘
            │
      对比上轮 Task Duration
            │
     ┌──────┴──────┐
     ▼             ▼
  改善 ≥ 5%    改善 < 5%
     │         连续 2 轮
     ▼             │
  下一轮       停止
```

## 日志归档

每轮优化的完整上下文自动归档到 `log/{OpName}/round_NNN/`，原始源码**只读不被修改**。

| 阶段 | 归档文件 | 负责者 |
|------|---------|--------|
| 基线 | `shapes.md` + `msprof/` + `perf_report.md` | Shaper |
| 策略 | `strategy.md` | Planner |
| 代码 | `op_kernel/` + `op_host/`（轮次内完整副本） | Coder |
| 性能 | `msprof/` + `perf_report.md` | Builder |
| 全局 | `summary.md` | 编排者 |

每轮目录包含完整可编译的算子工程副本（源码 + CMakeLists + scripts），以及 `build/`、`input/`、`output/`、`msprof/` 产物。

## Agent 知识体系

| Skill | 提供什么 | 使用者 |
|-------|---------|--------|
| `/env-check` | NPU/CANN/cmake/msprof 检查流程 | 编排者 |
| `/shape-design` | 三类 Shape 设计方法（边界/常见/压力） | Shaper |
| `/npu-architecture` | 芯片代际、UB 容量、峰值算力、条件编译宏 | 所有 Agent |
| `/perf-analysis` | CSV 指标解读、决策树、阈值速查 | Planner |
| `/perf-collection` | msprof 采集命令 + `perf_summary.py` 归档 | Shaper / Builder |
| `/precision-verify` | MERE/MARE 阈值标准 + verify_result.py 规范 | Shaper / Builder |
| `/kernel-patterns` | 8 种优化代码模式 + API 约束 + 编码红线 | Coder |
| `/error-diagnosis` | 编译/运行/精度三类诊断树 + 错误码速查 | Builder |

## 算子工程约定

每个算子工程遵循统一结构：

```
{OpName}/
├── CMakeLists.txt
├── op_host/
│   ├── {OpName}.asc          # Host 端：tiling 计算、ACL 内存管理
│   └── data_utils.h          # ReadFile/WriteFile 工具
├── op_kernel/
│   ├── {OpName}_kernel.asc   # Kernel：Init/Process/Compute/CopyIn/CopyOut
│   └── {OpName}_tiling.h     # TilingData 结构体 + 常量
└── scripts/
    ├── gen_data.py            # 生成输入数据和 golden
    ├── golden.py              # 参考实现
    └── verify_result.py       # 精度验证（MERE/MARE 阈值）
```

**Shape 定义**：通过命令行参数传入 `gen_data.py` 和 Host 可执行文件（如 `./Cummin B T F dim`），Shaper 在 `shapes.md` 中记录设计。

## 使用方式

```bash
# 1. 确保 NPU 环境就绪
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
source ${ASCEND_HOME_PATH}/set_env.sh

# 2. 在项目根目录启动 Claude Code
cd ascendc-optim-agent
claude

# 3. 指定算子工程启动优化
> 对 Cummin 算子执行优化流程
```

## 编码红线（Device 侧 op_kernel/）

- 禁止 `GlobalTensor::SetValue` / `GlobalTensor::GetValue` → 用 `DataCopyPad` 批量替代
- 禁止 `LocalTensor::SetValue` / `GetValue` 在热循环中逐元素操作（scan 类算子除外，无替代方案）
- 禁止硬编码 `blockDim` / UB 大小 / `blockIdx` → 用 `GetBlockIdx()` / TilingData 传递
- 用 `DataCopyPad` 替代 `DataCopy`（除非严格 32B 对齐）
- 禁止使用 `std::` 命名空间（Device 侧无 C++ 标准库；Host 端 `op_host/` 不受此限制）
- Ascend 910B2 禁止半精度标量运算（需 cast 到 float 比较）
- Ascend 910B2 上 `TBuf` 分配会导致 kernel launch 失败，避免使用

## 优化案例：Cummin（5 轮）

| 轮次 | 核心策略 | shape_0 | shape_1 | shape_2 | 备注 |
|------|---------|---------|---------|---------|------|
| 000 | 基线 | 5.14us | 724.96us | 22122.78us | |
| 001 | int64→int32 索引 | 4.98us | 692.12us | 21140.28us | -4.5% |
| 002 | 批量索引输出 | 5.18us | 343.08us | 9131.62us | **-58.7%** |
| 003 | 双缓冲+屏障合并 | 4.40us | 340.18us | 9142.26us | <1% |
| **004** | **智能 tileLen 负载均衡** | **3.92us** | **340.36us** | **5594.10us** | **-74.7%** |
| 005 | float runningMin+读写分离 | 4.58us | 357.82us | 5929.40us | +6% 退化 |

关键发现：Round 002 通过消除 IDX_BATCH 内循环将 MTE3 ratio 从 35%→8%；Round 004 通过 Host 端 tileLen 整除 F 消除多核负载不均衡（spread 从 7192us→21us）。scalar_ratio 达 93% 时为硬件极限。
