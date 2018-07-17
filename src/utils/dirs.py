# *- encoding: utf-8 -*-
"""
This file was created at Classy.org by Ben Cipollini, and taken from
https://github.com/classy-org/data-dives

Suite of methods for working with directories.
Multiple methods assume a data science project structure (see: https://drivendata.github.io/cookiecutter-data-science/)
The _data_science_dir function provides the basis for multiple functions - see its docstring for more details.
"""
import inspect
import os.path as op
import sys


def _get_caller_path(frames_above=0):
    """
    Returns the head of the path where the function is called.
    http://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python

    Note: requires being called by another function; don't call from global script.
    """
    assert len(inspect.stack()) > 2, 'Call _get_caller_path from a public function in the classypy.util.dirs package.'

    # frame[0] is this function, frame[1] is the internal function that
    # was called to get here, frame[2] is the caller,
    # frame[2 + frames_above] is for times when things are deeper.
    frame = inspect.stack()[2 + frames_above]  # 0 = this function, 1 = intermediate caller, 2 = actual caller
    caller_module_src_file = frame[1]  # frame is a tuple, 2nd value is the file
    caller_module_dir = op.dirname(op.abspath(caller_module_src_file))

    return caller_module_dir


def _repo_dir_and_children(path, max_levels=100):
    """
    Identifies the path from the root of the repo to
    `path'.

    Parameters
    ----------
    path : str
        Path to be checked.
    max_levels : int (default=100)
        Maximum levels to search in file tree for root.

    Returns
    -------
        Tuple of str: the path root and list: subdirs between path and root.
    """
    # Start from a path, and iterate until we find the repo root.
    path = op.abspath(path)
    children = []
    for li in range(max_levels + 1):  # protect against infinite loop
        if op.exists(op.join(path, '.git')) or op.exists(op.join(path, '.gitroot')):
            break
        if op.isdir(path):
            children.append(op.basename(path))
        path = op.dirname(path)

    if li <= max_levels:
        return path, children[::-1]
    else:
        return None, []


def _data_science_dir(path, dirname, base=None, subdir=None, max_levels=100):
    """
    Returns the path to `dirname`, assuming a data science organized repo.
    (see: https://drivendata.github.io/cookiecutter-data-science/)

    Parameters
    ----------
    path : str
        Path from which to search for `dirname`.
    dirname : str
        Name of data science dir to locate.
    base : str (default=None)
        (Optional) base directory for path.
    subdir : str (default=None)
        (Optional) directory under path.
    max_levels : int (default=100)
        Maximum levels to search in file tree for root.

    Returns
    -------
        str : File path `dirname` in path's repo.
    """
    _, children = _repo_dir_and_children(path, max_levels=max_levels)
    path = base_dir(path, base=base, max_levels=max_levels)
    if not path:
        return None

    # Check if exists; could be embedded
    new_path = path
    for child_dir in children[1:]:  # base_dir is children[0]
        new_path = op.join(new_path, child_dir)
        if not op.exists(new_path):  # doesn't exist anywhere, so go with raw
            break
        if op.exists(op.join(new_path, dirname)):
            path = new_path
            break
    path = op.join(path, dirname)

    if subdir:
        path = op.join(path, subdir)

    return path


def caller_dir(frames_above):
    return _get_caller_path(frames_above=frames_above)


def this_files_dir():
    # Go an extra frame deep, due to this function calling another function.
    return caller_dir(frames_above=1)


def base_dir(path=None, base=None, max_levels=100):
    """
    Returns the base directory for the given path, or path + base combo.
    If no path specified, returns the base directory relative to the
    location of the function call.

    Parameters
    ----------
    path : str (default=None)
        File path for which to identify base directory.
    base : str (default=None)
        Base directory of path.
    max_levels : int (default=100)
        Maximum levels to search in file tree for root.

    Returns
    -------
        str : File path of base directory.
    """
    path = path or _get_caller_path()
    path, children = _repo_dir_and_children(path, max_levels=max_levels)
    if path and base:
        # Explicit base
        return op.join(path, base)
    elif path and children:
        if children[0] in ['data', 'models', 'reports', 'src']:
            # The repo_dir IS the data science dir, so just return the repo_dir
            return path
        else:
            # Implicit base
            return op.join(path, children[0])
    else:
        # Not found
        return None


def dir_by_levels(path, levels):
    """
    Returns the file path corresponding to the
    dir `levels` levels above path.
    """
    return op.abspath(op.join(path, *(['..'] * levels)))


def repo_dir(path=None, max_levels=100):
    """
    Returns the head directory of the git repo
    containing `path`.

    Parameters
    ----------
    path : str (default=None)
        Path to be checked. If None, uses the path of the
        current working directory.
    max_levels : int (default=100)
        Maximum levels to search in file tree for root.

    Returns
    -------
        str : File name of path's head directory.
    """
    # Start from a path, and iterate until we find the repo root.
    path = path or _get_caller_path()
    path, children = _repo_dir_and_children(path, max_levels=max_levels)
    return path


def data_dir(path=None, base=None, subdir=None, max_levels=100):
    """
    Returns path to data directory in data science directory.
    """
    path = path or _get_caller_path()
    return _data_science_dir(
        path=path, dirname='data', base=base,
        subdir=subdir, max_levels=max_levels)


def models_dir(path=None, base=None, subdir=None, max_levels=100):
    """
    Returns path to models directory in data science directory.
    """
    path = path or _get_caller_path()
    return _data_science_dir(
        path=path, dirname='models', base=base,
        subdir=subdir, max_levels=max_levels)


def reports_dir(path=None, base=None, subdir=None, max_levels=100):
    """
    Returns path to reports directory in data science directory.
    """
    path = path or _get_caller_path()
    return _data_science_dir(
        path=path, dirname='reports', base=base,
        subdir=subdir, max_levels=max_levels)


def src_dir(path=None, base=None, subdir=None, max_levels=100):
    """
    Returns path to src directory in data science directory.
    """
    path = path or _get_caller_path()
    return _data_science_dir(
        path=path, dirname='src', base=base,
        subdir=subdir, max_levels=max_levels)


def add_to_path(src_dir=None, top=True):
    src_dir = src_dir or caller_dir(frames_above=1)
    if top:
        sys.path = [src_dir] + sys.path
    else:
        sys.path.append(src_dir)
