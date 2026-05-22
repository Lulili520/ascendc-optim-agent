# DiagPart 算子审查报告

**审查日期**: 2026-05-10
**审查轮次**: Round 1
**审查结论**: **FAIL**
**总分**: 55 / 100

---

## 审查结论

| 项目 | 结果 |
|------|------|
| **结论** | **FAIL** |
| **总分** | 55 / 100 |
| **结论原因** | 存在必须修复问题：入口属性缺少 `__aicore__`（检查项 2.2）、循环内使用 GetValue/SetValue（检查项 3.1） |

---

## 1. 独立构建验证 (Step 1)

### 1.1 CMake 配置验证

手动检查 CMakeLists.txt 各项要求：

| 检查项 | 结果 | 说明 |
|--------|------|------|
| `find_package(ASC REQUIRED)` | PASS | 第 3 行 |
| `LANGUAGES ASC CXX` | PASS | 第 5 行 `project(DiagPart LANGUAGES ASC CXX)` |
| `--npu-arch` | PASS | 第 30 行 `--npu-arch=dav-2201` |
| 链接 `tiling_api` | PASS | 第 15 行 |
| `target_include_directories` | PASS | 包含 op_kernel 和 op_host |

### 1.2 独立编译

| 项目 | 结果 |
|------|------|
| cmake 配置 | PASS |
| make 编译 | PASS（无错误、无警告） |
| 可执行文件生成 | PASS (`build/DiagPart`) |

---

## 2. 代码质量评估 (Step 2)

### 2.1 维度 1：编译验证 (10/10)

| 检查项 | 分值 | 结果 |
|--------|------|------|
| 1.1 独立编译成功 | 7 | 7/7 |
| 1.2 无代码级警告 | 3 | 3/3 |

### 2.2 维度 2：架构合规 (9/15)

| 检查项 | 分值 | 结果 | 说明 |
|--------|------|------|------|
| 2.1 TPipe/TQue 模式 | 3 | 3/3 | 使用 TPipe + TQue 架构，模式正确 |
| 2.2 入口属性正确 | 3 | 1/3 | **问题**：入口函数声明为 `__global__ __vector__`，缺少 `__aicore__` 属性。标准应为 `__aicore__ __global__`。当前 `__vector__` 在部分平台上可能工作，但不符合标准 Ascend C Kernel 入口属性要求 |
| 2.3 定义顺序正确 | 3 | 3/3 | Kernel 类在前，入口函数在后 |
| 2.4 内存管理配对 | 0 | 0/3 | **问题**：`outLocal` 在第 46 行通过 `AllocTensor` 分配，但只在 `outQueue.FreeTensor` 时释放。中间的 `CopyInAndExtract` 函数内对 `inQueueX` 的 AllocTensor/FreeTensor 配对正确，但 `outLocal` 的生命周期跨多个 sub-tile 循环迭代，AllocTensor 后直到 CopyOut 的 FreeTensor 之间没有问题。然而，DOUBLE_BUFFER 设为 1，队列深度为 1，无法实现双缓冲流水线 |
| 2.5 数据流完整 | 2 | 2/3 | 数据流基本完整：CopyIn -> Extract -> CopyOut。但 GetValue/SetValue 不经过 TQue 队列，属于标量操作，数据流中断 |

### 2.3 维度 3：编码规范 (7/15)

| 检查项 | 分值 | 结果 | 说明 |
|--------|------|------|------|
| 3.1 矢量 API | 0 | 0/4 | **必须修复**：第 72-74 行在循环内使用 `GetValue`/`SetValue` 逐元素操作。这是标量操作，性能极低。应使用矢量 API 或 Gather/Duplicate 等批量操作提取对角线元素 |
| 3.2 API 约束满足 | 4 | 4/4 | DataCopyPad 使用正确，stride 参数和 pad 参数配置合理 |
| 3.3 数据对齐 | 3 | 3/4 | SUB_TILE=16 对 half 类型满足 32 字节对齐。DataCopyPad 本身处理非对齐场景 |
| 3.4 命名规范 | 0 | 0/3 | 命名不符合 `{功能}_custom` 规范（如 `KernelDiagPart_custom`），类名和函数名均缺少 `_custom` 后缀 |

**GetValue/SetValue 问题详情**（第 72-74 行）：
```cpp
for (uint32_t i = 0; i < subCount; i++) {
    outLocal.SetValue(s * SUB_TILE + i, xLocalDeq.GetValue(i * (SUB_TILE + 1)));
}
```
此循环每次迭代执行一次 GM 读取（GetValue）和一次 UB 写入（SetValue），属于 `GlobalTensor::GetValue/SetValue` 黑名单 API，效率极低。应替换为矢量批量操作。

### 2.4 维度 4：性能优化 (6/20)

| 检查项 | 分值 | 结果 | 说明 |
|--------|------|------|------|
| 4.1 动态硬件参数 | 4 | 4/4 | 核数通过 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取，TILE_LENGTH/SUB_TILE 为编译常量但属于合理范围 |
| 4.2 多核并行 | 2 | 2/4 | 多核切分逻辑存在，但 N=128 时仅使用 1 核，N=512 时仅 2 核。负载基本均衡（两核耗时 16.99us vs 16.53us，差异 2.8% < 10%），但利用率低 |
| 4.3 流水线/双缓冲 | 0 | 0/4 | **问题**：`DOUBLE_BUFFER = 1`，队列深度为 1，完全无法实现搬运/计算重叠。性能数据证实：MTE2 占比 62.9%，VEC 占比仅 0.09%，scalar 占比 33.5%，是典型的搬运密集+标量瓶颈模式，无流水线重叠效果 |
| 4.4 同步策略 | 0 | 0/4 | **问题**：无 PipeBarrier 调用（总数=0），但代码中 DataCopyPad(MTE2) 后紧跟 GetValue(Scalar)，存在 MTE2->Scalar 的 RAW 跨 pipe 依赖，缺少必要的同步。当前依赖 TQue 的 EnQue/DeQue 隐式同步来保证 inQueueX 的数据可见性，但 `outLocal` 不经过队列直接用 SetValue 写入，FreeTensor 后的 DataCopyPad 读 `outLocal` 可能存在数据竞争风险 |
| 4.5 计算效率与上板性能 | 0 | 0/4 | **问题**：Task Duration 17.5us（N=512），理论搬运量 = 输入 512KB + 输出 1KB，理论搬运耗时约 0.28us（按 1.8TB/s 带宽），实际耗时 17.5us，差距超过 60 倍。主要瓶颈：循环内逐元素 GetValue/SetValue 标量操作占比 33.5%，极大拉低性能 |

#### 性能数据摘要（N=512, 独立采集）

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration | 17.50 us | - |
| Block Dim | 2 | 低（910B 可用 20-40 核） |
| Current Freq / Rated Freq | 1800 / 1800 | 满频运行 |
| aiv_mte2_ratio (core0/core1) | 62.9% / 60.4% | MTE2 占主导 |
| aiv_scalar_ratio (core0/core1) | 33.5% / 36.1% | 标量占比过高 |
| aiv_vec_ratio | 0.09% | VEC 几乎未使用 |
| aiv_mte3_ratio (core0/core1) | 0.82% / 1.86% | 输出量小 |
| GM_to_UB datas | 32 KB / core | 搬运量合理 |
| UB_to_GM datas | 0.5 KB / core | 输出量极小 |
| aiv_vec_total_cflt_ratio | 0% | 无 bank conflict |
| L2 Cache total hit rate | 6.4% / 7.5% | 极低（但数据量小，正常） |
| aiv_vec_wait_ratio | 62.7% / 61.1% | 等待占比高 |
| aiv_mte2_active_bw | 2.85 / 3.06 GB/s | 远低于峰值带宽 |

**性能瓶颈分析**：
1. **标量操作瓶颈**：GetValue/SetValue 循环导致 scalar 占比高达 33-36%
2. **无流水线重叠**：DOUBLE_BUFFER=1，MTE2 搬运与标量提取完全串行
3. **搬运粒度极小**：每次仅搬运 16x16 half = 512B 子矩阵，远小于推荐的 16KB 最优搬运粒度
4. **带宽利用率极低**：MTE2 active bandwidth 仅 2.85-3.06 GB/s，远低于峰值 1.8 TB/s

### 2.5 维度 5：测试覆盖 (12/15)

| 检查项 | 分值 | 结果 | 说明 |
|--------|------|------|------|
| 5.1 测试数据生成 | 4 | 4/4 | gen_data.py 支持参数化 N 值，生成随机正态分布数据 |
| 5.2 结果验证脚本 | 4 | 4/4 | verify_result.py 使用 np.allclose 对比，输出 max diff 和 mismatch count |
| 5.3 Level 0 覆盖 | 4 | 4/4 | N=128 (128元素) 和 N=512 (512元素) 均覆盖 |
| 5.4 精度标准明确 | 0 | 0/3 | **问题**：verify_result.py 中 rtol=1e-3, atol=1e-3 对 FP16 正确，但脚本中未区分 dtype，如果未来扩展 dtype 需要修改。且缺少 Level 2（极值/零值边界）测试用例 |

**测试级别覆盖**：

| 测试级别 | 要求 | 覆盖情况 |
|---------|------|---------|
| Level 0 (8-16 元素) | 必须 | 未覆盖（最小 N=128） |
| Level 1 (1K 元素) | 推荐 | 未覆盖（N=512 仅 512 元素） |
| Level 2 (极值/零值) | 推荐 | 未覆盖 |
| Level 3 (大数据量) | 可选 | 未覆盖 |

**说明**：虽然 N=128 和 N=512 的测试验证了核心功能，但缺少小规模 Level 0 测试（如 N=4 或 N=16）和边界测试。

### 2.6 维度 6：精度验证 (6/10)

| 检查项 | 分值 | 结果 | 说明 |
|--------|------|------|------|
| 6.1 FP32 全用例 PASS | N/A | N/A | 算子仅支持 FP16，不适用 |
| 6.2 FP16 全用例 PASS | 3 | 3/3 | N=128: bit-exact (max diff=0.0), N=512: bit-exact (max diff=0.0) |
| 6.3 BF16 全用例 PASS | N/A | N/A | 算子未实现 BF16，不适用 |

**精度独立验证结果**（Reviewer 独立运行）：

| Shape | dtype | max_diff | 结果 |
|-------|-------|----------|------|
| [128, 128] | FP16 | 0.0 | PASS (bit-exact) |
| [512, 512] | FP16 | 0.0 | PASS (bit-exact) |

注：由于算子仅做数据搬运（提取对角线），无数值计算，bit-exact 结果符合预期。扣除 4 分是因为仅支持 FP16 单一 dtype。

### 2.7 维度 7：文档 (5/15)

| 检查项 | 分值 | 结果 | 说明 |
|--------|------|------|------|
| 7.1 README.md 存在 | 0 | 0/3 | **缺失**：项目根目录无 README.md |
| 7.2 数学公式 | 3 | 3/3 | DESIGN.md 包含 `out[i] = x[i, i]` 公式 |
| 7.3 编译运行指南 | 2 | 2/3 | run.sh 存在，但无独立编译运行文档说明 |
| 7.4 API 映射/约束 | 0 | 0/3 | 无 API 映射表和约束说明 |
| 7.5 已知限制 | 0 | 0/3 | 无已知限制说明 |

---

## 3. 必须修复问题列表

### M1: 入口属性不规范 [检查项 2.2]

**位置**: `op_kernel/DiagPart_kernel.asc` 第 106 行
**当前代码**:
```cpp
extern "C" __global__ __vector__ void DiagPart_kernel(GM_ADDR x, GM_ADDR out, GM_ADDR tiling)
```
**问题**: 使用 `__vector__` 而非标准 `__aicore__` 属性。标准 Ascend C Kernel 入口应使用 `__aicore__`。
**修复建议**: 改为 `extern "C" __global__ __aicore__ void DiagPart_kernel(...)` 或确认当前平台对 `__vector__` 的支持策略。

### M2: 循环内使用 GetValue/SetValue 黑名单 API [检查项 3.1]

**位置**: `op_kernel/DiagPart_kernel.asc` 第 72-74 行
**当前代码**:
```cpp
for (uint32_t i = 0; i < subCount; i++) {
    outLocal.SetValue(s * SUB_TILE + i, xLocalDeq.GetValue(i * (SUB_TILE + 1)));
}
```
**问题**: GetValue/SetValue 属于 API 黑名单，效率极低。性能数据证实 scalar 占比高达 33-36%。
**修复建议**: 使用矢量 Gather 或 Duplicate 等批量操作替代。或者使用 DataCopyPad 配合 stride 参数直接将子矩阵对角线元素搬运到连续位置，避免逐元素标量操作。

### M3: 同步缺失导致潜在数据竞争 [检查项 4.4]

**位置**: `op_kernel/DiagPart_kernel.asc` 第 69-74 行
**问题**: `DataCopyPad`（MTE2 pipe）写入 `xLocal` 后，立即通过 `GetValue`（Scalar pipe）读取 `xLocalDeq`。虽然 TQue 的 EnQue/DeQue 提供隐式同步，但 `outLocal` 通过 SetValue（Scalar pipe）写入后，最终通过 `DataCopyPad`（MTE3 pipe）读取 `outLocal`（第 85 行），存在 Scalar->MTE3 跨 pipe RAW 依赖，缺少同步保障。
**修复建议**: 在 `outQueue.EnQue(outLocal)` 后、`CopyOut` 中的 `DeQue` 后，确认 TQue 的 EnQue/DeQue 是否已提供足够同步。若 TQue 隐式同步已覆盖，则说明同步充分；否则需在 SetValue 写完后、DataCopyPad 读之前添加 `PipeBarrier<PIPE_ALL>`。

### M4: DOUBLE_BUFFER=1 无法实现双缓冲 [检查项 4.3]

**位置**: `op_kernel/DiagPart_tiling.h` 第 7 行
**当前代码**:
```cpp
constexpr uint32_t DOUBLE_BUFFER = 1;
```
**问题**: 队列深度设为 1，完全无法实现搬运/计算流水线重叠。
**修复建议**: 改为 `constexpr uint32_t DOUBLE_BUFFER = 2;` 并调整 buffer 大小使 UB 总占用不超过限制。

---

## 4. 建议优化项（非阻塞）

### S1: 扩大 SUB_TILE 提高搬运粒度

当前 SUB_TILE=16，每次仅搬运 512B。建议增大到 64 或 128，提高 MTE2 带宽利用率。

### S2: 添加 Level 0 测试用例

建议添加 N=4 或 N=16 的小规模测试用例，覆盖 Level 0 基础功能验证。

### S3: 创建 README.md

项目缺少 README.md 文档，建议补充算子概述、编译运行指南、测试结果说明。

### S4: 添加 FP32/BF16 数据类型支持

当前仅支持 FP16，建议扩展支持 FP32 和 BF16。

### S5: 使用矢量 API 替代标量提取

建议研究使用 `AscendC::Gather` 或其他矢量 API 替代 GetValue/SetValue 逐元素操作，从根本上消除标量瓶颈。

---

## 5. 同步策略逐项依赖分析

**代码中无显式 PipeBarrier 调用**（总数=0）。

依赖 TQue EnQue/DeQue 的隐式同步。逐项分析：

| 序号 | 前操作 | 前 Pipe | 后操作 | 后 Pipe | 依赖类型 | 判定 |
|------|--------|---------|--------|---------|---------|------|
| 1 | DataCopyPad(GM->UB, xLocal) | MTE2 | EnQue(xLocal) | TQue同步 | MTE2->TQue | 隐式同步覆盖 |
| 2 | DeQue(xLocalDeq) | TQue同步 | GetValue(xLocalDeq) | Scalar | TQue->Scalar | EnQue/DeQue 隐式同步 |
| 3 | SetValue(outLocal) | Scalar | EnQue(outLocal) | TQue同步 | Scalar->TQue | 需确认 TQue 是否提供 Scalar 同步 |
| 4 | DeQue(outLocal) | TQue同步 | DataCopyPad(UB->GM) | MTE3 | TQue->MTE3 | 隐式同步覆盖 |

**风险点**：第 3 项 SetValue(Scalar pipe) -> EnQue(TQue) 的同步保障不明确。TQue 的 EnQue 通常同步 VECTOR pipe 的写入，但 SetValue 属于 Scalar pipe 操作。如果 TQue 不保障 Scalar pipe 的数据可见性，则存在数据竞争。

**冗余率**：N/A（无显式 PipeBarrier，冗余率不适用）。
**同步充分性评分**：由于依赖隐式同步且存在 Scalar pipe 同步不确定性，评 0/4。

---

## 6. 硬件参数硬编码检查

| 检查项 | Grep 命令 | 结果 |
|--------|----------|------|
| blockDim 硬编码 | `grep -n "blockDim\s*=\s*[0-9]"` | PASS - 未发现 |
| blockIdx 硬编码 | `grep -n "blockIdx\s*=\s*[0-9]"` | PASS - 未发现 |
| 核数动态获取 | Host 代码 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` | PASS |

---

## 7. 评分汇总

| 维度 | 满分 | 得分 | 关键扣分项 |
|------|------|------|-----------|
| 1. 编译验证 | 10 | 10 | - |
| 2. 架构合规 | 15 | 9 | 入口属性 -2, 内存管理配对 -3, 数据流 -1 |
| 3. 编码规范 | 15 | 7 | GetValue/SetValue 黑名单 -4, 命名规范 -3, 对齐 -1 |
| 4. 性能优化 | 20 | 6 | 双缓冲失效 -4, 同步风险 -4, 标量瓶颈性能差 -4, 多核利用低 -2 |
| 5. 测试覆盖 | 15 | 12 | 精度标准不够灵活 -3 |
| 6. 精度验证 | 10 | 6 | 仅 FP16 单一 dtype -4 |
| 7. 文档 | 15 | 5 | 无 README -3, 无 API 映射 -3, 无限制说明 -3, 运行指南不完整 -1 |
| **总计** | **100** | **55** | - |

---

## 8. 最终判定

| 项目 | 值 |
|------|------|
| **结论** | **FAIL** |
| **总分** | 55 / 100 |
| **必须修复问题数** | 4 |
| **建议优化项数** | 5 |

**判定理由**：
1. 总分 55 < 70，不满足 PASS 或 PASS WITH NOTES 门槛
2. 存在检查项 2.2（入口属性）和 3.1（GetValue/SetValue 黑名单 API）的必须修复问题
3. 性能严重不达标：Task Duration 17.5us vs 理论 0.28us，差距超过 60 倍
4. 双缓冲完全失效（DOUBLE_BUFFER=1），流水线无重叠

**Developer 修复优先级**：
1. **最高优先级**：替换 GetValue/SetValue 为矢量批量操作 (M2)
2. **高优先级**：设置 DOUBLE_BUFFER=2 启用双缓冲 (M4)
3. **高优先级**：确认并修复 Scalar->TQue 同步保障 (M3)
4. **中优先级**：修正入口属性为标准 `__aicore__` (M1)
