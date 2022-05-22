#!/usr/bin/env python3
__author__ = 'Martin Hochwallner <marthoch@users.noreply.github.com>'
__email__ = "marthoch@users.noreply.github.com"
__license__ = "BSD 3-clause"

# import os
import numpy as np
# import datetime
# import pandas as pd
# from numpy import ndarray
import matplotlib.pyplot as plt

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def read_line_from_frame(img, line=None, hline=None, vline=None, interpolation='linear', plot_show_line=False):
    if plot_show_line:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img - 273.15, interpolation='nearest', cmap='plasma')
        ax.plot([line['p0']['h'], ], [line['p0']['v'], ], color='lime', lw=2, alpha=0.5, marker='o',
                markersize=20)
        ax.plot([line['p1']['h'], ], [line['p1']['v'], ], color='deepskyblue', lw=2, alpha=0.5,
                marker='s', markersize=20)
        ax.plot([line['p0']['h'], line['p1']['h']], [line['p0']['v'], line['p1']['v']],
                color='lime', lw=1, alpha=0.8)
        return fig

    if hline is not None:
        if (hline % 1) == 0.0:
            val = img[int(hline), :]
            v = h = None
            return dict(v=v, h=h, values=val)
        else:
            if interpolation == 'linear':
                a = (hline % 1)
                l = int(hline)
                val = (1 - a) * img[l, :] + a * img[int(l + 1), :]
                v = h = None
                return dict(v=v, h=h, values=val)
            elif interpolation == 'nearest':
                l = int(round(hline))
                v = h = None
                return dict(v=v, h=h, values=img[l, :])
            else:
                raise Exception("interpolation not known: use interpolation='linear' or 'nearest'")
    if vline is not None:
        if (vline % 1) == 0.0:
            return img[:, int(vline)]
        else:
            if interpolation == 'linear':
                a = (vline % 1)
                l = int(vline)
                return (1 - a) * img[:, int(vline)] + a * img[:, int(vline + 1)]
            elif interpolation == 'nearest':
                l = int(round(vline))
                v = h = None
                return dict(v=v, h=h, values=img[:, l])
            else:
                raise Exception("interpolation not known: use interpolation='linear' or 'nearest'")
    if line is not None:
        lineLen = line.get('len')
        lineLenX = max(abs(line['p1']['v'] - line['p0']['v']), abs(line['p1']['h'] - line['p0']['h'])) + 1
        if lineLen is None:
            lineLen = lineLenX
        if lineLen < lineLenX:
            raise Exception("down sampling interpolation is not implemented")
        v = np.linspace(line['p0']['v'], line['p1']['v'], lineLen)
        h = np.linspace(line['p0']['h'], line['p1']['h'], lineLen)
        if interpolation == 'linear':
            vi = v.astype(int)
            hi = h.astype(int)
            va = v - vi
            ha = h - hi
            val = (1 - va) * (1 - ha) * img[vi, hi] + va * (1 - ha) * img[vi + 1, hi] + (1 - va) * ha * img[
                vi, hi + 1] + va * ha * img[vi + 1, hi + 1]
        elif interpolation == 'nearest':
            vi = v.round(decimals=0).astype(int)
            hi = h.round(decimals=0).astype(int)
            val = img[vi, hi]
        else:
            raise Exception("interpolation not known: use interpolation='linear' or 'nearest'")
        return dict(v=v, h=h, values=val)
    raise Exception('not enough parameters')


def read_line_from_arrayofframes(frames, start=0, stop=None, step=1, line=None, hline=None, vline=None, interpolation='linear', plot_show_line=False):

    if plot_show_line:
        return read_line_from_frame(frames[:,:,start], line=line, hline=hline, vline=vline, interpolation=interpolation, plot_show_line=plot_show_line)

    stop_ = stop if stop is not None else frames.shape[2]
    n_frames = (stop_ - start) // step
    fi = np.empty(n_frames, dtype=np.uint32)
    aline = read_line_from_frame(frames[:, :, start], line=line, hline=hline, vline=vline, interpolation=interpolation)
    v = np.empty([aline['values'].shape[0], n_frames], dtype=np.float32)

    for i, frameNr in enumerate(range(start, stop_, step)):
        # display(i)
        fi[i] = frameNr
        v[:, i] = read_line_from_frame (frames[:, :, frameNr],  line=line, hline=hline, vline=vline, interpolation=interpolation)['values']

    return dict(values=v, frame_number=fi)

# eof
