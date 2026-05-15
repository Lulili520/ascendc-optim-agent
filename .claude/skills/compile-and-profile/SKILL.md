---
name: compile-and-profile
description: 算子编译、精度验证与 msprof 性能采集归档流程。
---

# 编译、运行与性能采集

## 执行流程

```
Step 1: 环境配置 → source set_env.sh
Step 2: 编译 + 精度验证 → bash run.sh
    失败 → 报告错误，不进入 Step 3
    通过 ↓
Step 3: msprof op 性能采集
Step 4: 归档到 docs/perf/round_NNN/
```

## Step 1：环境配置

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
which cms && which msprof
```

## Step 2：编译与精度验证

```bash
cd {op_name}
bash run.sh   # 运行全部用例
```

`run.sh` 内部：source set_env.sh → cmake + make → gen_data → 执行 Kernel → verify_result。

## Step 3：msprof op 性能采集

**必须从算子工程根目录执行。**

`<shape_idx>` 为 `run.sh` 中 CASE 数组的下标（0, 1, 2...），host 端查找表自动映射为实际 shape 参数。

```bash
cd {op_name}
source ${ASCEND_HOME_PATH}/set_env.sh

msprof op \
    --warm-up=10 \
    --output=./msprof_output \
    ./build/{op_name} <shape_idx>
```

| 参数 | 说明 |
|------|------|
| `--warm-up=10` | 预热 10 次，避免 DVFS 降频干扰 |
| `--output=./msprof_output` | 固定输出目录 |
| `./build/{op} <shape_idx>` | 可执行文件 + shape 索引（位置参数，非 --application/--application-args） |

## Step 4：归档

`perf_summary.py` 位于 skill 目录 `.claude/skills/compile-and-profile/scripts/perf_summary.py`，用法：

```
python3 perf_summary.py <opprof_dir> <ops_dir>
```

从算子工程根目录执行：

```bash
cd {op_name}

# msprof 输出在 ./msprof_output/OPPROF_{timestamp}_XXX/ 下
OPPROF_DIR=$(ls -d ./msprof_output/OPPROF_* 2>/dev/null | tail -1)
python3 ../.claude/skills/compile-and-profile/scripts/perf_summary.py "$OPPROF_DIR" .
```

脚本自动完成：找下一个 round_NNN → 拷贝所有 CSV → 生成 summary.txt。

归档后：`docs/perf/round_NNN/` 含 8 CSV + summary.txt。

## 精度阈值

| 数据类型 | rtol | atol |
|---------|------|------|
| FP16 | 1e-3 | 1e-4 |
| FP32 | 1e-5 | 1e-6 |
| INT | — | 0 |

## 规则

1. 精度验证从**算子工程根目录**执行，直接 `bash run.sh` 运行全部用例
2. 精度验证失败 → 中止本轮，不发散到性能采集
3. 性能采集固定 `--output=./msprof_output`，不可用其他名称
4. round 编号由 perf_summary.py 管理，禁止手动创建/跳号
5. 编译错误原样上报，不自行修复
