from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="lowlevel_ops",
    ext_modules=[
        CppExtension(
            name="lowlevel_ops._C",
            sources=["cpu/vector_add.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
