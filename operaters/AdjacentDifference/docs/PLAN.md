# AdjacentDifference 算子开发计划

## Phase 0: 环境检查 ✅
- NPU: 910B2, CANN 8.5.0, msprof 可用

## Phase 1: 方案设计 ✅
- DESIGN.md 完成
- Tiling: 8192 元素/UB tile, 多核均匀切分
- Buffer: 6 个 UB buffer, 总计 ~136KB

## Phase 2: 算子开发

### 文件清单
| 文件 | 说明 |
|------|------|
| op_kernel/AdjacentDifference_tiling.h | Tiling 常量与结构体 |
| op_kernel/AdjacentDifference_kernel.asc | Kernel 实现 |
| op_host/AdjacentDifference.asc | Host 入口（ACL 初始化 + Tiling 计算 + Kernel 启动） |
| op_host/data_utils.h | 二进制 I/O |
| CMakeLists.txt | 构建配置 |
| run.sh | 一键构建+测试 |
| scripts/golden.py | Golden 计算 |
| scripts/gen_data.py | 测试数据生成 |
| scripts/verify_result.py | 精度验证 |

### 开发步骤
1. 创建 tiling.h（常量 + TilingData 结构体）
2. 创建 kernel.asc（Init + CopyIn/Compute/CopyOut 三阶段流水线）
3. 创建 host .asc（ACL 初始化 + Tiling 计算 + KernelCall）
4. 创建 CMakeLists.txt（可执行目标）
5. 创建测试脚本（gen_data.py + golden.py + verify_result.py）
6. 创建 run.sh
7. 编译并运行
8. 精度验证

## Phase 3: 代码审查
- 使用 ascendc-code-review skill 审查
- 输出 REVIEW.md

## Phase 4: 性能采集
- 使用 msprof 采集性能数据
- 分析并优化
