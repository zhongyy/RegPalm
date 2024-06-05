import setuptools

#with open("README.rst", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="acctools",
    version="1.1.0",
    author="anonymous",
    author_email="anonymous",
    description="Tool to make test easy, including roc, \
        hard example mining, feature reader, data list generator, and etc.",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #python_requires='>=3.6',
    #data_files=['examples/paint/DejaVuSans.ttf'],
)