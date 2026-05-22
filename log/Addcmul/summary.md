# Addcmul 优化全局摘要

## 算子信息
- 算子: Addcmul
- 计算: `output = self + value * (tensor1 * tensor2)`
- 数据类型: FP16 输入输出，FP32 计算
- NPU: Ascend 910B2

## 优化轮次: 1 轮

### Round 001 优化内容
- **P1**: Double Buffer（TQue 1→2 buffer）+ UB_TILE 4096→8192 + t1 预取流水线
- **P2**: Bank conflict padding（inQueue +128 字节）

### 性能对比（Baseline → Round 001）

| shape | 名称 | Baseline(us) | Optimized(us) | 改善(%) |
|-------|------|-------------|---------------|---------|
| 0 | common_large [8,2048,4096] | 527.20 | 383.84 | **27.20%** |
| 1 | common_small [1,1024,4096] | 32.56 | 20.10 | **38.26%** |
| 2 | boundary_tail [1,512,1001] | 7.48 | 5.96 | **20.32%** |

### 关键指标变化
- MTE2 带宽利用率: 6.96% → 98.77%（common_large）
- vec_ratio: 29.87% → 32.09%（common_large）
- mte2_wait: 99.4% → 显著降低（双缓冲重叠）

## 结论
一轮优化即实现 20-38% 性能提升。核心优化为双缓冲+增大搬运粒度+预取流水线，有效解决了 MTE2 带宽瓶颈。
