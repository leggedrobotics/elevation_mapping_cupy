from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["semantic_sensor",], install_requires=["torch", "torchvision",], package_dir={"": "script"},
)

setup(**setup_args)
