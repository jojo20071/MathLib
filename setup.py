from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='MathLib',
  version='0.0.1',
  description='A very basic MathLib',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Johannes Häderle',
  author_email='johanneshaederle18092@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='Bezier', 
  packages=find_packages(),
  install_requires=[''] 
)