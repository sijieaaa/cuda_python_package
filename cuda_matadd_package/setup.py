from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_torch_extension",
    ext_modules=[
        CUDAExtension(
            name="cuda_torch_extension",
            sources=["cuda_torch_extension.cu"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
