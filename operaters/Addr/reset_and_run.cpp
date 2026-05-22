#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main() {
    setbuf(stdout, NULL);
    printf("Step 1: aclInit...\n");
    aclError ret = aclInit(nullptr);
    printf("aclInit: %d\n", ret);

    printf("Step 2: aclrtSetDevice(0)...\n");
    ret = aclrtSetDevice(0);
    printf("aclrtSetDevice(0): %d\n", ret);

    if (ret != ACL_SUCCESS) {
        printf("FATAL: cannot set device\n");
        aclFinalize();
        return 1;
    }

    int64_t maxCores = 0;
    ret = aclrtGetDeviceInfo(0, ACL_DEV_ATTR_VECTOR_CORE_NUM, &maxCores);
    printf("Step 3: aclrtGetDeviceInfo: ret=%d, maxCores=%ld\n", ret, maxCores);

    printf("Step 4: Resetting device...\n");
    ret = aclrtResetDevice(0);
    printf("aclrtResetDevice(0): %d\n", ret);

    aclFinalize();
    printf("Step 5: aclFinalize done. Now re-initializing...\n");

    ret = aclInit(nullptr);
    printf("Step 6: aclInit again: %d\n", ret);

    ret = aclrtSetDevice(0);
    printf("Step 7: aclrtSetDevice(0) again: %d\n", ret);

    if (ret == ACL_SUCCESS) {
        aclrtResetDevice(0);
    }
    aclFinalize();
    printf("DONE\n");
    return 0;
}
