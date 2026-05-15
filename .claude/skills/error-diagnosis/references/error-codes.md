# AscendC 常见错误码

## 运行时错误码

| 错误码 | 名称 | 常见原因 |
|--------|------|---------|
| 507035 | VEC core exception | DataCopyPad 非 32B 对齐、UB 溢出、越界访问 |
| 507034 | MTE exception | GM 地址越界、非对齐访问 |
| 207001 | aclrtMemcpy failed | Host 端内存分配失败或大小不匹配 |
| 107000 | Device not available | NPU 驱动未加载或被占用 |

## SOC 错误码

| 错误码 | 说明 |
|--------|------|
| 0xff | 通用硬件异常 |
| 0xfe | 指令非法 |
