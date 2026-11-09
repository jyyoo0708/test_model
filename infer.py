#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run local 96x96 inference with either system python or python3."""
from __future__ import print_function
import os
import sys


def main():
    root = os.path.dirname(os.path.realpath(__file__))
    executable = os.path.join(root, '.venv', 'bin', 'python')
    script = os.path.join(root, 'scripts', 'infer_images96.py')
    if not os.path.isfile(executable):
        raise SystemExit('Local runtime missing: ' + executable)
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TMPDIR'] = os.path.join(root, 'tmp')
    env['XDG_CACHE_HOME'] = os.path.join(root, '.cache')
    env['TORCH_HOME'] = os.path.join(root, '.cache', 'torch')
    if not os.path.isdir(env['TMPDIR']):
        os.makedirs(env['TMPDIR'])
    os.execve(executable, [executable, script] + sys.argv[1:], env)


if __name__ == '__main__':
    main()
