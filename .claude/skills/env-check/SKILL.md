---
name: env-check
description: AscendC 算子优化环境检查。验证 NPU 设备可用、CANN 环境就绪、编译器工具链和 msprof 工具完整。
---

# 优化环境检查

## 概述

在进入优化流程之前，必须通过本环境检查。**任一项失败 → 终止。**

## 检查流程

```
Step 1: NPU 设备检查 → npu-smi info
Step 2: CANN 环境检查 → ASCEND_HOME_PATH + set_env.sh
Step 3: 编译工具链检查 → cmake + make
Step 4: 性能采集工具检查 → msprof
全部通过 → 进入优化流程
```

## Step 1：NPU 设备检查

```bash
npu-smi info
```
判定：列出至少一个 NPU 设备，状态正常。

## Step 2：CANN 环境检查

```bash
echo "${ASCEND_HOME_PATH}"
ls "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/set_env.sh"
```
判定：`ASCEND_HOME_PATH` 非空，`set_env.sh` 存在且可执行。

## Step 3：编译工具链检查

```bash
which cmake && cmake --version
which make && make --version
```

## Step 4：性能采集工具检查

```bash
which msprof && msprof --help
```

## 通过标准

四项全部通过后输出环境概要：

| 检查项 | 结果 |
|--------|------|
| NPU 设备 | Ascend 910B2 |
| CANN 环境 | ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0 |
| cmake | 版本号 |
| msprof | available |

## 常见问题

| 问题 | 排查方向 |
|------|---------|
| NPU 不可见 | `npu-smi info` 是否识别设备；检查驱动 |
| 算子运行失败 | 优先运行 `source set_env.sh`；检查 `ASCEND_OPP_PATH` |
| 编译找不到头文件 | 确认 CMakeLists.txt 中 `target_include_directories` 包含 `ASCEND_HOME_PATH/opp` |

## 环境变量注意事项

- 使用 `ASCEND_HOME_PATH`（不是 `ASCEND_HOME`）
- 每次新 shell 会话必须重新 `source set_env.sh`
- 编译和 msprof 命令必须在 `source set_env.sh` 之后执行
