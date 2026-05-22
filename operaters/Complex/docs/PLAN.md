# Complex Operator Implementation Plan

## Step 1: Create Project Structure
- Copy template from add_custom
- Rename all references: add_custom → Complex
- Set up directory layout

## Step 2: Implement Kernel Layer (op_kernel/)
- `Complex_tiling.h`: Tiling constants (TILE_LENGTH=4096) and ComplexTilingData struct
- `Complex_kernel.asc`: KernelComplex class with CopyIn→Compute→CopyOut pipeline
  - 2 input queues (real, imag), 1 output queue (interleaved)
  - Compute: block-interleave using DataCopy local-to-local (BLOCK=32)

## Step 3: Implement Host Layer (op_host/)
- `Complex.asc`: Host main entry with ACL init, tiling calculation, kernel launch
  - Input: input_real.bin, input_imag.bin → Output: output.bin
  - Total elements: 4 * 2048 * 2048 = 16,777,216 per input
  - Output: 33,554,432 elements
- `data_utils.h`: Copy from template (unchanged)

## Step 4: Implement PyTorch Extension (op_extension/)
- `Complex_torch.cpp`: PyTorch host with tiling + kernel launch
- `register.cpp`: TORCH_LIBRARY registration for nppu.Complex
- `ops.h`: Function declarations

## Step 5: Implement Test Scripts (scripts/)
- `gen_data.py`: Generate fp16 random data for shape [4, 2048, 2048]
- `golden.py`: Block-interleaved golden computation
- `verify_result.py`: Compare output vs golden with fp16 tolerances
- `test_torch.py`: PyTorch pathway verification

## Step 6: Build Configuration
- `CMakeLists.txt`: Dual target (executable + shared library), npu-arch=dav-2201
- `run.sh`: Build/run/verify script

## Step 7: Compile and Verify
- Build the project
- Run gen_data.py to generate test data
- Execute the kernel
- Verify output against golden

## Step 8: Code Review and Performance
- Generate REVIEW.md
- Fix any review issues
- Run performance profiling
