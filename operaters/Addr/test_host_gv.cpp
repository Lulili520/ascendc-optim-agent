#include <cstdio>
#include "acl/acl.h"

// No kernel include - testing plain ACL init
extern "C" void test_kernel(GM_ADDR x1, GM_ADDR y);
extern "C" void test_kernel_gv(GM_ADDR x1, GM_ADDR x2, GM_ADDR y);

int main() {
    setbuf(stdout, NULL);
    printf("1. aclInit...\n");
    aclInit(nullptr);
    printf("2. aclrtSetDevice(0)...\n");
    auto ret = aclrtSetDevice(0);
    printf("3. aclrtSetDevice=%d\n", ret);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return 1;
    }
    aclrtResetDevice(0);
    aclFinalize();
    printf("4. Done\n");
    return 0;
}
