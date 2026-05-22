#!/bin/bash
# Reset device then run Addr from correct directory
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# Reset device first
/root/autodl-tmp/ll/cannbot-skills/ascendc_code/Addr/reset_device

# Change to build directory so relative paths work
cd /root/autodl-tmp/ll/cannbot-skills/ascendc_code/Addr/build

# Generate test data if needed
if [ ! -f input/input_x1_case1.bin ]; then
    python3 ../scripts/gen_data.py 512 1024
fi

# Create output dir
mkdir -p output

# Run with case 1
./Addr 512 1024

echo "EXIT CODE: $?"
