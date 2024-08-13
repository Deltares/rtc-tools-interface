from setuptools import find_packages, setup

import versioneer


setup(
    name="rtc-tools-interface",
    version=versioneer.get_version(),
    maintainer="Deltares",
    packages=find_packages("."),
    author="Deltares",
    description="Toolbox for user interfaces for RTC-Tools",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "pydantic",
        "casadi != 3.6.6",
        "rtc-tools >= 2.7.0a3",
    ],
    tests_require=["pytest", "pytest-runner"],
    python_requires=">=3.9",
    cmdclass=versioneer.get_cmdclass(),
)
