#include <cstdint>
#include <cstdio>
#include "acl/acl.h"

int main() {
    setbuf(stdout, NULL);
    printf("A\n");
    aclInit(nullptr);
    printf("B\n");
    int32_t deviceId = 0;
    auto ret = aclrtSetDevice(deviceId);
    printf("C: aclrtSetDevice=%d\n", ret);
    if (ret == ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
    }
    aclFinalize();
    printf("D\n");
    return 0;
}
