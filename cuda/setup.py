from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='segment_mm',
    ext_modules=[
        CUDAExtension('segment_mm', [
            'segment_mm.cpp',
            'segment_mm_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })