from setuptools import find_packages, setup

import versioneer


setup(
    name='rtc-tools-interface',
    version=versioneer.get_version(),
    maintainer='Deltares',
    packages=find_packages("."),
    author='Deltares',
    description="Toolbox for user interfaces for RTC-Tools",
    install_requires=["pandas",
                      "rtc-tools >= 2.5.0",
                      "matplotlib",
                      "numpy",
                      "pydantic"],
    tests_require=['pytest', 'pytest-runner'],
    python_requires='>=3.5',
    cmdclass=versioneer.get_cmdclass(),
)
