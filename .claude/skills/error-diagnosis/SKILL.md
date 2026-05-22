---
name: error-diagnosis
description: AscendC 算子错误诊断与修复指南。覆盖编译错误、运行时崩溃、精度不达标三类问题。
---

# AscendC 算子错误诊断与修复

## 诊断流程

```
收到错误报告
    ↓
读错误日志（第一条 + 最后一条）
    ↓
diff op_kernel/{op}_kernel.asc op_kernel/{op}_kernel.asc.bak
    ↓
┌────┴────────────────────────────┐
│ 编译错误    运行时崩溃    精度异常 │
▼             ▼             ▼
编译诊断树   运行诊断树    精度诊断树
```

## 编译错误速查

| 错误信息 | 根因 | 修复 |
|---------|------|------|
| `undefined reference to 'xxx'` | 缺少库链接 | 检查 target_link_libraries |
| `no matching function for DataCopy` | 张量类型/维度不匹配 | 确认 LocalTensor 类型与 API 参数一致 |
| `UB buffer size exceeds limit` | 总 UB 分配超容量 | 减小 UB_FORMER 或减少 BUFFER_NUM；检查 Σ(BUF_NUM_i × per_buf_i) < UB_CAP |
| `unknown type name 'half'` | 缺少头文件 | 确保 `#include "kernel_operator.h"` |
| `__global__ function overloaded` | kernel 入口重复 | 确保函数名唯一 |

## 运行时错误速查

| 错误/信号 | 根因 | 修复 |
|---------|------|------|
| `SIGSEGV` | 越界内存访问 | 检查 `xGm[tileIdx * ubFormer]` 索引不越界 |
| `device hang` / 超时 | 死循环或 EnQue/DeQue 不配对 | 检查 EnQue/DeQue + AllocTensor/FreeTensor 配对 |
| `aclrtMemcpy failed` | Host 端内存错误 | 检查 inputByteSize/outputByteSize |
| `507035` (向量核异常) | DataCopyPad 对齐或 UB 溢出 | 检查非对齐尾部处理；验证 UB 总用量 |

## 精度错误速查

| 现象 | 根因 | 修复 |
|------|------|------|
| 输出全零 | CopyOut 未执行或 EnQue/DeQue 错序 | 检查 Process() 循环逻辑 |
| 部分元素错误 | tail 处理逻辑错误 | 检查 tailElementNum 和 tail 分支 |
| 随机元素错误 | UB 未初始化或同步问题 | 确保 AllocTensor 后数据完整写入 |
| MERE/MARE 超阈值 | FP16 累积误差 | 考虑混合精度：FP16→FP32 计算→FP16 |
| 每次结果不同 | 未初始化的 UB buffer | 写入前零初始化或确保全覆盖 |

## 精度阈值

精度标准详见 `/precision-verify` skill。核心规则：

- **浮点输出**：MERE < Threshold 且 MARE < 10×Threshold（FP16: 2^-10, FP32: 2^-13, BF16: 2^-7）
- **整数/索引输出**：二进制一致
- **禁止**使用 `np.allclose`

## 诊断步骤

### 编译错误
1. 定位第一个错误行 → 交叉对照 kernel 源码
2. 检查 include 路径（CMakeLists.txt 中 target_include_directories）
3. 验证 API 参数类型 — 查阅 kernel-patterns 的 API 约束
4. Tiling 相关错误：检查 TilingData 字段名与 kernel 引用一致

### 运行时错误
1. `export ASCEND_SLOG_PRINT_TO_STDOUT=1` 启用详细日志
2. 检查 EnQue/DeQue 配对 + AllocTensor/FreeTensor 配对
3. 检查 tileIdx 范围：确保 `i < tileNum`

### 精度错误
1. 用最小 shape 复现，缩小排查范围
2. 在 Compute() 中临时添加 PipeBarrier 排除同步问题
3. 对比 `.bak` 版本与当前版本输出差异
4. 检查 FP16 中间值是否超出表示范围（±65504）

## 规则

1. 修复根因而非表象
2. 保留优化意图，不能为了修复而撤销优化
3. 一次修复一个错误，修后交 Builder 验证
4. 保留 `.bak` 文件作为正确行为参照
5. 无法定位时用 WebSearch 搜索 Ascend 社区

## 参考文档

| 文档 | 内容 |
|------|------|
| [错误码速查](references/error-codes.md) | 运行时错误码与 SOC 错误码 |
| [精度调试流程](references/precision-debug.md) | 二分法定位、逐元素对比、常见精度问题 |
