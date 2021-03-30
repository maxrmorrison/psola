from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='psola',
    description='Time-domain pitch-synchronous overlap-add',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/psola',
    install_requires=['numpy',
                      'pypar',
                      'soundfile',
                      'tqdm',
                      'praat-parselmouth'],
    packages=['psola'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'duration', 'pitch', 'speech', 'stretch', 'vocode'],
    license='GPLv3')
