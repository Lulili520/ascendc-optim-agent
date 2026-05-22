# Addr 算子开发计划

## 文件清单

| 文件 | 说明 |
|------|------|
| op_kernel/Addr_tiling.h | Tiling 常量与数据结构 |
| op_kernel/Addr_kernel.asc | Kernel 实现（行优先处理 + FP32 混合精度） |
| op_host/Addr.asc | Host 入口（ACL 初始化、Tiling 计算、Kernel 调度） |
| op_host/data_utils.h | 文件读写工具函数 |
| op_extension/Addr_torch.cpp | PyTorch binding |
| op_extension/register.cpp | TORCH_LIBRARY 注册 |
| op_extension/ops.h | 函数声明 |
| scripts/golden.py | Golden 参考计算 |
| scripts/gen_data.py | 生成测试数据 |
| scripts/verify_result.py | 精度验证 |
| scripts/test_torch.py | PyTorch 路径测试 |
| scripts/mare_mere_threshold.py | MERE/MARE 精度阈值工具 |
| CMakeLists.txt | 双目标构建（可执行文件 + .so） |
| run.sh | 一键构建运行脚本 |

## 开发步骤

1. 创建项目目录结构
2. 实现 Tiling 头文件（UB_FORMER=8192, 行优先切分）
3. 实现 Kernel（行优先处理，FP32 混合精度：Muls + Add）
4. 实现 Host 入口（读取 M/N 参数和 alpha/beta 标量）
5. 实现 PyTorch 扩展
6. 编写测试脚本（双测试用例：[512,1024] 和 [1024,2048]）
7. 编译构建
8. 精度验证

## 验证计划

- 直调路径：run.sh 0 (case1 [512,1024]) + run.sh 1 (case2 [1024,2048])
- PyTorch 路径：test_torch.py 覆盖两组 shape
- 精度标准：MERE < 2^(-10), MARE < 10 * 2^(-10)
