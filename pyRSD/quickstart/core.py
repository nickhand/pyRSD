import os
from six.moves.urllib import request
import tarfile
import argparse

EXAMPLES = ['galaxy/survey-poles', 'galaxy/periodic-poles', 'galaxy/periodic-pkmu']

def download_data(dirname, example):
    """
    Download the pyRSD-data github tarball to the specified directory

    Parameters
    ----------
    dirname : str
        the output directory
    example : str
        the path of the example to download
    """
    assert example in EXAMPLES

    # make the cache dir if it doesnt exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # download the tarball locally
    tarball_link = "https://codeload.github.com/nickhand/pyRSD-data/legacy.tar.gz/master"
    tarball_local = os.path.join(dirname, 'master.tar.gz')
    request.urlretrieve(tarball_link, tarball_local)

    if not tarfile.is_tarfile(tarball_local):
        dir_exists = os.path.exists(os.path.dirname(tarball_local))
        args = (tarball_local, str(dir_exists))
        raise ValueError("downloaded tarball '%s' cannot be opened as a tar.gz file (directory exists: %s)" %args)

    # extract the tarball to the cache dir
    with tarfile.open(tarball_local, 'r:*') as tar:

        members = tar.getmembers()
        topdir = members[0].name
        basepath = os.path.join(topdir, example)

        for m in members[1:]:
            if example in m.name:
                name = os.path.relpath(m.name, basepath)
                m.name = name
                tar.extract(m, path=dirname)

    # remove the downloaded tarball file
    if os.path.exists(tarball_local):
        os.remove(tarball_local)


def main():

    desc = "download example data and parameter files to get up and running with pyRSD"
    parser = argparse.ArgumentParser(description=desc)

    h = "which example to download"
    parser.add_argument('example', choices=EXAMPLES, help=h)

    h = "the output directory to save the downloaded files; if it doesn't exist it will be created"
    parser.add_argument('dirname', type=str, help=h)

    ns = parser.parse_args()

    # first download the files
    download_data(ns.dirname, ns.example)

    # update DIRNAME in the params file
    param_file = os.path.join(ns.dirname, 'params.dat')
    params = open(param_file, 'r').read()
    params = params.replace("$(DIRNAME)", os.path.abspath(ns.dirname))
    with open(param_file, 'wb') as ff:
        ff.write(params.encode())
