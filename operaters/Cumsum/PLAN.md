# Cumsum 算子开发计划

## 文件清单

| 文件 | 说明 |
|------|------|
| op_kernel/Cumsum_tiling.h | Tiling 数据结构 |
| op_kernel/Cumsum_kernel.asc | Kernel 实现 |
| op_host/Cumsum.asc | Host 入口（含 ACL 初始化、Kernel 启动） |
| op_host/data_utils.h | 文件读写工具 |
| scripts/golden.py | Golden 参考实现 |
| scripts/gen_data.py | 测试数据生成 |
| scripts/verify_result.py | 精度验证 |
| CMakeLists.txt | 构建配置 |
| run.sh | 一键运行脚本 |

## 开发步骤

1. 创建 tiling 数据结构 (Cumsum_tiling.h)
2. 实现 Kernel (Cumsum_kernel.asc) — dim=0/1 向量化 Add，dim=2 FP32 顺序累加
3. 实现 Host 入口 (Cumsum.asc) — Tiling 计算、Kernel 调用
4. 编写测试脚本 (golden.py, gen_data.py, verify_result.py)
5. 编译验证 + 精度测试
6. 代码审查
