from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

this_dir = Path(__file__).resolve().parent

sources = [
    str(this_dir / "csrc" / "bindings.cpp"),
    str(this_dir / "csrc" / "metal" / "powerfoam_metal.mm"),
]

ext_modules = [
    CppExtension(
        name="torch_powerfoam_metal._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc")],
        extra_compile_args=["-std=c++17", "-fobjc-arc"],
        extra_link_args=["-framework", "Foundation", "-framework", "Metal"],
    )
]

setup(
    name="torch-powerfoam-metal",
    version="0.1.0",
    description="Torch-native Metal forward rasterizer for bounded power-cell foam",
    packages=["torch_powerfoam_metal"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
