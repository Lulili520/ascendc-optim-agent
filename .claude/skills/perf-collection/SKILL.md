---
name: perf-collection
description: NPU 性能采集与归档。msprof op 采集 + perf_summary.py 归档到指定目录 + 生成 summary.txt。
---

# NPU 性能采集与归档

## 采集流程

```
Step 1: 确认编译通过 + 精度验证通过
    ↓
Step 2: msprof op 采集（warm-up=10）
    ↓
Step 3: perf_summary.py 归档 + 生成 summary.txt
    ↓
Step 4: 读 summary.txt 获取关键指标
```

## 前置条件

- 算子已编译通过（`build/{op_name}` 存在）
- 精度验证已通过（所有 shape PASS）
- CANN 环境已加载

**精度未通过时禁止进入性能采集。**

## Step 1：环境确认

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
which msprof || { echo "msprof not found"; exit 1; }
```

## Step 2：msprof op 采集

在轮次目录中执行。**每个 shape 采集前必须先调 gen_data.py 生成对应输入数据**（算子从文件读取输入）：

```bash
cd log/{op}/round_NNN

# 1. 生成当前 shape 的输入数据
python3 scripts/gen_data.py <param1> <param2> ...

# 2. 采集
msprof op \
    --warm-up=10 \
    --output=./msprof_output \
    ./build/{op_name} <param1> <param2> ...
```

| 参数 | 说明 |
|------|------|
| `--warm-up=10` | 预热避免 DVFS 降频，始终使用 |
| `--output=./msprof_output` | 固定临时输出目录 |
| `./build/{op} <params>` | 算子二进制 + shape 参数 |

输出 `./msprof_output/OPPROF_{timestamp}/` 含 8 个 CSV。

## Step 3：归档

```bash
cd log/{op}/round_NNN

OPPROF_DIR=$(ls -d ./msprof_output/OPPROF_* 2>/dev/null | tail -1)
python3 ../../.claude/skills/perf-collection/scripts/perf_summary.py "$OPPROF_DIR" ./msprof/shape_<N>
```

脚本将 CSV 复制到 `./msprof/shape_<N>/` 并生成 `summary.txt`。

## Step 4：读取关键指标

从 `summary.txt` 提取供性能分析使用：

| 指标 | 位置 | 用途 |
|------|------|------|
| Task Duration | OpBasicInfo | 总耗时，与上轮对比 |
| Block Dim | OpBasicInfo | 多核启用判定 |
| scalar / vec / mte2 / mte3 ratio | PipeUtilization | 瓶颈类型判定 |
| icache_miss_rate | PipeUtilization | 指令缓存判定 |

> 详细阈值和判定标准见 `/perf-analysis`，本 skill 只负责采集和归档。

## 多 shape 批量采集

```bash
cd log/{op}/round_NNN
source ${ASCEND_HOME_PATH}/set_env.sh

# shapes 从上轮 shapes.md 获取
N=0
for params in "<param1_0> <param2_0> ..." "<param1_1> <param2_1> ..." "..."; do
    python3 scripts/gen_data.py $params
    msprof op --warm-up=10 --output=./msprof_output ./build/{op_name} $params
    OPPROF_DIR=$(ls -d ./msprof_output/OPPROF_* | tail -1)
    python3 ../../.claude/skills/perf-collection/scripts/perf_summary.py "$OPPROF_DIR" ./msprof/shape_$N
    N=$((N + 1))
done
```

## 规则

1. 精度未通过 → 禁止采集
2. 固定 `--warm-up=10`、`--output=./msprof_output`
3. shape 编号由调用方管理，perf_summary.py 直接归档到指定目录
4. 退化如实记录，不追加补救
5. 每次采集后立即归档
