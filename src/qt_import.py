#!/usr/bin/env python
# ecnoding: utf-8

'''
Handler for standardized Qt4/Qt5 module importing
'''

__author__      = 'Jake Retallick'
__copyright__   = 'Apache License 2.0'
__version__     = '1.2'
__date__        = '2018-05-07'

from importlib import import_module

#_qt_mods = ['PySide', 'PySide2', 'PyQt5', 'PyQt4']
_qt_mods = ['PySide', 'PySide2']

maps = {}
maps['PySide'] = {'QtWidgets': 'QtGui', 'QtPrintSupport': 'QtGui'}
maps['PySide2'] = {}
#maps['PyQt4'], maps['PyQt5'] = maps['PySide'], maps['PySide2']

def _remap(submod):
    mp = maps[qt]
    return mp[submod] if submod in mp else submod

def _get_qt():
    '''Returns the latest QtX module'''

    for m in _qt_mods:
        try:
            import_module(m)
        except ImportError:
            continue
        break
    else:
        print('One of [{0}] must be installed'.format(' '.join(_qt_mods)))
        m = None

    return m

def import_qt_attr(submod, attr):
    '''Return a reference to the given attribute of QtX.submod

    usage:
        QSvgGenerator = import_qt_attr('QtSvg', 'QSvgGenerator')
    '''

    if qt is None:
        return None

    try:
        sm = import_module('{0}.{1}'.format(qt, _remap(submod)))
        return getattr(sm, attr)
    except ImportError:
        print('Failed to load submodule: {0}'.format(submod))
    except AttributeError:
        print('Invalid attribute: {0}.{1}'.format(sm.__name__, attr))

    return None


def import_qt_mod(*submods, **kwargs):
    '''Returns a list of the request submodule for QtX for your latest
    local version of Qt. Input should either be a list of submodule names or
    a position based set of *args. Missing/invalid submodules will return None
    at that sire in the list.

    usage:
        QtGui, QtCore = import_qt_mod('QtGui', 'QtCore')
        QtGui, QtCore, QtSvg = import_qt_mod(['QtGui', 'QtCore', 'QtSvg'])

    wildcard usage:
        import_qt_mod('QtGui', 'QtCore', wc=globals()) is equivalent to
        from QtX.QtGui import *
        from QtX.QtCore import *
    '''

    if len(submods)==1 and isinstance(submods[0], list):
        submods = submods[0]

    if qt is None:
        return [None for _ in submods]

    # load all submodules
    out = []
    for submod in submods:
        try:
            sm = import_module('{0}.{1}'.format(qt, _remap(submod)))
            if 'wc' in kwargs and kwargs['wc'] is not None:
                for k, v in sm.__dict__.items():
                    if isinstance(k, str) and not k.startswith('_'):
                        kwargs['wc'][k] = v
        except ImportError:
            print('Failed to import submodule: {0}'.format(submod))
            sm = None
        out.append(sm)

    return out[0] if len(out)==1 else out

qt = _get_qt()
