from setuptools import setup, find_packages

setup(
    name="elevation_mapping_cupy",
    version="0.0.1",
    packages=find_packages(where="script"),
    package_dir={"": "script"},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your_email@example.com",
    description="Description of the elevation_mapping_cupy package.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 'node_name = elevation_mapping_cupy.some_module:main_function',
        ],
    },
)


# from distutils.core import setup
# from catkin_pkg.python_setup import generate_distutils_setup
# from setuptools import find_packages
# setup_args = generate_distutils_setup(
#     packages=["elevation_mapping_cupy", "elevation_mapping_cupy.plugins","elevation_mapping_cupy.kernels","elevation_mapping_cupy.fusion"], package_dir={"": "script"},
    
# )

# setup(**setup_args)


