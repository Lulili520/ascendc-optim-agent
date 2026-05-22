#include "acl/acl.h"
#include <cstdio>

int main() {
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);

    // Try reset first
    ret = aclrtResetDevice(0);
    printf("aclrtResetDevice(0): %d\n", ret);

    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice(0): %d\n", ret);

    if (ret == ACL_SUCCESS) {
        ret = aclrtResetDevice(0);
        printf("aclrtResetDevice(0) after set: %d\n", ret);
    }

    aclFinalize();
    return 0;
}
