#! /usr/bin/env python
#
# Utils file
#
# @author: Koen Eppenhof
# @email: k.a.j.eppenhof@tue.nl
# @date: 2018/08/30


import numpy
DTYPE = numpy.float64


def dfm_opts(dfm):
    return {
        'cmap': 'bwr',
        'vmin': -max(dfm.min(), dfm.max()),
        'vmax': max(dfm.min(), dfm.max())
    }


def dfm_show(dfm):
    return {
        'X' : dfm,
        'cmap': 'bwr',
        'vmin': -max(dfm.min(), dfm.max()),
        'vmax': max(dfm.min(), dfm.max())
    }
