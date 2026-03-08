import os
import subprocess
import torch # 必須導入以抓取路徑
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def get_nvcc_cuda_version(cuda_home):
    """檢查 nvcc 的版本"""
    if not cuda_home:
        cuda_home = "/usr/local/cuda"
    
    nvcc_bin = os.path.join(cuda_home, 'bin', 'nvcc')
    try:
        output = subprocess.check_output([nvcc_bin, "--version"], universal_newlines=True)
        for line in output.split('\n'):
            if "release" in line:
                ver_str = line.split("release")[-1].split(",")[0].strip()
                return float(ver_str)
    except Exception:
        return 0.0
    return 0.0

# 1. 取得路徑資訊
current_cuda_ver = get_nvcc_cuda_version(CUDA_HOME)
# 自動定位 PyTorch 庫路徑 (解決你 find 到的那個長路徑)
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
cuda_lib_path = os.path.join(CUDA_HOME, "lib")

print(f"Detected CUDA Version: {current_cuda_ver}")
print(f"Torch lib path: {torch_lib_path}")
# setup.py 建議修正
# setup.py 建議修正
nvcc_flags = [
    "-O3",
    "-allow-unsupported-compiler",
    # 移除所有手動 -D_Float 定義，避免重複定義衝突
]

setup(
    name="density_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="density_cuda_ext",
            sources=["density_cuda.cu"],
            extra_compile_args={
                # 在 CXX 部分明確指定標準，有助於 nvcc 協調標頭檔
                "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": nvcc_flags,
            },
            extra_link_args=[
                "/ISPD26-Contest/submit/lib/libcudart.so",
                os.path.join(torch_lib_path, "libc10_cuda.so"),
                os.path.join(torch_lib_path, "libtorch_cuda.so"),
                f"-Wl,-rpath,{torch_lib_path}",
                f"-Wl,-rpath,/ISPD26-Contest/submit/lib"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)