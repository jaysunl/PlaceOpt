Build instructions (manual)

1) From the repo root:
   cd src/sta/density_ext

2) Build in place:
   python setup.py build_ext --inplace

The build will emit a shared library (density_cuda_ext*.so) in this folder.
The Python code in src/sta/density.py will automatically use it when available.

Notes:
- Requires nvcc and a compatible PyTorch CUDA build.
- The extension currently supports float32 CUDA tensors.
