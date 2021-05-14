import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Seg_Bact_TLS",
    version="0.0.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="Segmentation of bacteria growing in agar-pads, imaged by transmitted light (stacks)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/seg_bact_TL",
    download_url = 'https://github.com/jeanollion/seg_bact_TL/archive/v_01.tar.gz',
    packages=setuptools.find_packages(),
    keywords = ['Segmentation', 'Bacteria', 'Transmitted Light', 'Microscopy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow', 'keras_preprocessing', 'edt==2.0.2', 'dataset_iterator>=0.2.1']
)
