import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pyRSD', parent_package, top_path)

    config.add_subpackage('cosmology')
    config.add_subpackage('data')
    config.add_subpackage('rsd')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)