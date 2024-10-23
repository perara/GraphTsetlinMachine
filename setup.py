from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import sysconfig


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Get Python include path
        python_include = sysconfig.get_path('include')

        # Get nanobind path
        nanobind_path = subprocess.check_output(['python3', '-m', 'nanobind', '--cmake_dir']).decode('utf-8').strip()
        nanobind_path = os.path.dirname(os.path.dirname(nanobind_path))  # Get the grandparent directory

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYTHON_INCLUDE_DIR=' + python_include,
            '-DNANOBIND_PATH=' + nanobind_path
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', os.path.abspath('bindings')] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='GraphTsetlinMachine',
    version='0.3.1',
    author='Ole-Christoffer Granmo',
    author_email='ole.granmo@uia.no',
    url='https://github.com/cair/GraphTsetlinMachine/',
    license='MIT',
    description='Graph Tsetlin Machine',
    long_description='Graph Tsetlin Machine for Deep Logical Learning and Reasoning',
    keywords='pattern-recognition cuda machine-learning interpretable-machine-learning rule-based-machine-learning propositional-logic graph tsetlin-machine regression convolution classification multi-layer',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numba',
        'scikit-image',
        'tensorflow',
        'pycuda',
        'numpy',
        'nanobind'
    ],
    ext_modules=[CMakeExtension('GraphTsetlinMachine.graphtsetlinmachine')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.6',
)