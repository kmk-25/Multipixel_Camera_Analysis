from setuptools import setup

setup(
    name='multipixel_camera_analysis',

    url='https://github.com/kmk-25/Multipixel_Camera_Analysis',
    author='Kenneth Kohn',
    author_email='kmkohn@stanford.edu',

    py_modules=['multipixel_camera_analysis'],
    install_requires=[
    'numpy','scipy','matplotlib','h5py'],
)