#!/usr/bin/env python3
__author__ = 'Martin Hochwallner <marthoch@users.noreply.github.com>'
__email__ = "marthoch@users.noreply.github.com"
__license__ = "BSD 3-clause"

import os
import numpy as np
import datetime
import pandas as pd
from numpy import ndarray
#import matplotlib.pyplot as plt

from . import helper

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

try:
    import fnv
    import fnv.reduce
    import fnv.file
    # documentation file:///C:/Program%20Files/FLIR%20Systems/sdks/file/python/doc/index.html
except Exception as e:
    print("""
## Installation of the fnv module:
* Download and install the "FLIR Science File SDK" from https://flir.custhelp.com/app/account/fl_download_software.
* Install Python module according https://flir.custhelp.com/app/answers/detail/a_id/3504/~/getting-started-with-flir-science-file-sdk-for-python  
    Attention: Use the setup.py attached to the "Getting started with ..."  as the one in the SDK is broken.        
""")
    raise e


class SEQfileReader:
    """SEQfileReader: High level reader for FLIR .seq files (thermography recordings (IR))
    based on the FLIR Science File SDK

>>> seq = SEQfileReader(filename=r'path to file.seq')

>>> seq
SEQfile(filename=r'testRecording.seq')
    unit: ............... TEMPERATURE_FACTORY / KELVIN
    image size: ......... 640 x 120
    number of frames: ... 199
    recording start time: 2000-00-00T00:00:00.000000
    recording time: ..... 1 sec
    frame rate: ......... 200.0 Hz
    current frame:
        frame number: ....... 0
        preset number: ...... 1
        date/time of frame:   2000-00-00T00:00:00.000000

    """

    frameRate_nominal_available = 200. / 2 ** np.arange(0, 5)

    def __init__(self, filename):
        log.debug('opening "%s"', filename)
        self._filename = filename
        self.im = fnv.file.ImagerFile(self._filename)
        self.im.unit = fnv.Unit.TEMPERATURE_FACTORY
        self.im.temp_type = fnv.TempType.KELVIN
        self.im.get_frame(0)
        self._firstFrame_time = self.im.frame_info.time

        # total recording time, frame rate
        self.im.get_frame(self.im.num_frames - 1)
        last_frame_time = self.im.frame_info.time
        fr_ = 1 / ((last_frame_time - self._firstFrame_time).total_seconds() / (self.im.num_frames - 1))
        idx = (np.abs(self.frameRate_nominal_available - fr_)).argmin()
        self.frameRate = self.frameRate_nominal_available[idx]
        self.recordingTime = last_frame_time - self._firstFrame_time

        # reset
        self.im.get_frame(0)

        self.timeArray = None

    @property
    def filename_full(self):
        return self._filename

    @property
    def filename(self):
        return os.path.basename(self._filename)

    @property
    def filename_woext(self):
        """Filename without extension
        """
        return self.filename.rsplit('.', 1)[0]

    @property
    def number_of_frames(self):
        """ Number of Frames in the SEQ file
        """
        return self.im.num_frames

    @property
    def first_frame_time(self):
        return self._firstFrame_time

    @property
    def current_frame_number(self):
        return self.im.frame_number

    @property
    def current_frame_time(self):
        return self.im.frame_info.time

    @property
    def emissivity(self):
        return self.im.reduce_objects.get_object_parameters().emissivity

    @property
    def atmospheric_transmission(self):
        return self.im.reduce_objects.get_object_parameters().atmospheric_transmission

    def go2next_frame(self, step=1):
        try:
            self.go2frame(self.current_frame_number + step)
        except IndexError:
            raise StopIteration

    def go2frame(self, frame_number):
        """
        :param frame_number: uses python notation, 0 is first, -1 last
        :return: nothing

        """
        if frame_number < 0:
            frame_number = self.number_of_frames + frame_number
        if 0 > frame_number or frame_number >= self.im.num_frames:
            raise IndexError('Out of range: {} not it [0..{}]'.format(frame_number, self.im.num_frames - 1))
        else:  # return value of get_frame is not reliable
            if not self.im.get_frame(frame_number):
                self.im.get_frame(0)
                raise IndexError('Out of range: {} not it [0..{}]'.format(frame_number, self.im.num_frames - 1))

    def frame_iter(self, start=0, stop=None, step=1):
        return FrameIterator(seq=self, start=start, stop=stop, step=step)

    def __str__(self):
        return """SEQfileReader(filename=r'{s.filename}')
    unit: ............... {s.im.unit.name} / {s.im.temp_type.name} 
    image size: ......... {s.im.width} x {s.im.height}
    number of frames: ... {s.number_of_frames}
    recording start time: {recStartTime}
    recording time: ..... {recordingTime} sec
    frame rate: ......... {s.frameRate} Hz
    current frame:
        frame number: ....... {s.current_frame_number}
        preset number: ...... {s.im.num_presets}
        date/time of frame:   {frame_info_time}
        emissivity:               {s.emissivity}
        atmospheric_transmission: {s.atmospheric_transmission}
        """.format(s=self,
                   frame_info_time=self.im.frame_info.time.isoformat(),
                   recStartTime=self._firstFrame_time.isoformat(),
                   recordingTime=self.recordingTime.total_seconds())

    __repr__ = __str__

    def get_frame(self):
        return np.array(self.im.final, copy=False).reshape((self.im.height, self.im.width))

    get_image = get_frame  # backwards compatibility, shall not be used

    def set_active_frame(self, fn):
        return self.im.get_frame(fn)

    def get_time_df_nom(self):
        """Time vector nominally spaced (based on frame rate) as pandas DataFrame."""
        return pd.DataFrame({'time_nom': self._firstFrame_time + datetime.timedelta(days=0, seconds=(
                1 / self.frameRate)) * np.arange(self.im.num_frames)})

    def get_time_sec0_df_nom(self):
        """Time vector in sec starting with 0, nominally spaced (based on frame rate) as pandas DataFrame."""
        return pd.DataFrame({'timeSec0_nom': (1 / self.frameRate) * np.arange(self.im.num_frames)})

    def read_time_df(self, reread=False):
        if reread or (self.timeArray is None):
            t = np.empty(self.im.num_frames, dtype=datetime.datetime)
            fi: ndarray = np.empty(self.im.num_frames, dtype=np.uint32)
            for i, fr in enumerate(self.im.frame_iter(0)):
                t[i] = fr.frame_info.time
                fi[i] = fr.frame_number
            self.timeArray = pd.DataFrame({'time_file': t, 'fn': fi})
        return self.timeArray

    def read_as_numpy(self, selection=None, frame_range=None):
        """Read as 3d numpy array:

        :param selection:   [[row0, row1],[col0, col1]] ... [[row from to},[col from to]]
        :param frame_range: [startFrame, endFrame]

        The return format corresponds with scikit-image: (pln, row, col)
        https://scikit-image.org/docs/stable/user_guide/numpy_images.html#notes-on-the-order-of-array-dimensions
        """
        if frame_range is None:
            n_frames = self.im.num_frames
            frame_range = [0, self.im.num_frames]
        else:
            n_frames = frame_range[1] - frame_range[0]
        if selection is None:
            n_row = self.im.height
            n_col = self.im.width
        else:
            n_row = selection[0][1] - selection[0][0]
            n_col = selection[1][1] - selection[1][0]

        # preallocate arrays
        fi = np.empty(n_frames, dtype=np.uint32)
        t = np.empty(n_frames, dtype=datetime.datetime)
        v = np.empty([n_frames, n_row, n_col, ], dtype=np.float32)

        # for fr in self.im.frame_iter(0):
        for i, fn in enumerate(range(*frame_range)):
            self.im.get_frame(fn)
            fr = self.im
            fi[i] = fr.frame_number
            t[i] = fr.frame_info.time
            img = np.array(fr.final, copy=False).reshape((fr.height, fr.width))
            if selection is None:
                v[i, :, :] = img
            else:
                v[i, :, :] = img[selection[0][0]:selection[0][1], selection[1][0]:selection[1][1]]
        if frame_range is None:  # when reading all
            self.timeArray = pd.DataFrame({'time': t})
        return t, v, fi

    def read_line(self, line=None, hline=None, vline=None, interpolation='linear', plot_show_line=False):
        """
        line = dict(p0=dict(v=1, h=1), p1=dict(v=10, h=50), len=None)
        interpolation='linear' or 'nearest'
        :param plot_show_line: if True plot the image and show the line
        """
        img = self.get_frame()
        # img = np.ones([100,100])
        # for i in np.arange(0,100):
        #    for j in np.arange(0,100):
        #        img[i,j] = i
        return helper.read_line_from_frame(img, line=line, hline=hline, vline=vline, interpolation=interpolation, plot_show_line=plot_show_line)


    def read_line_over_time(self, start=0, stop=None, step=1, line=None, hline=None, vline=None,
                            interpolation='linear'):
        """

        :param start:
        :param stop:
        :param step:
        :param line:  dict(p0=dict(v=1, h=1), p1=dict(v=10, h=50), len=None)
        :param hline:
        :param vline:
        :param interpolation:  'linear' or 'nearest'
        :return:
        """

        # preallocate arrays
        stop_ = stop if stop is not None else self.number_of_frames
        n_frames = (stop_ - start) // step
        fi = np.empty(n_frames, dtype=np.uint32)
        t = np.empty(n_frames, dtype=datetime.datetime)
        self.go2frame(0)
        aline = self.read_line(line=line, hline=hline, vline=vline, interpolation=interpolation)
        v = np.empty([aline['values'].shape[0], n_frames], dtype=np.float32)

        for i, seqFrame in enumerate(self.frame_iter(start=start, stop=stop, step=step)):
            # display(i)
            fi[i] = seqFrame.current_frame_number
            t[i] = seqFrame.current_frame_time
            v[:, i] = self.read_line(line=line, hline=hline, vline=vline, interpolation=interpolation)['values']

        return dict(time=t, values=v, frame_number=fi)


class FrameIterator:
    """Iterator over Frames
    It acts on the SEQ's current frame thus there can be only one iterator active at the same time.
    """

    def __init__(self, seq, start=0, stop=None, step=1):
        self.seq = seq
        self.start = start
        self.stop = stop
        self.step = step
        self.firstIter = None

    def __iter__(self):
        self.seq.go2frame(self.start)
        self.firstIter = True
        return self

    def __next__(self):
        try:
            if self.firstIter:
                self.firstIter = False
                return self.seq
            if (self.stop is None) or (self.seq.current_frame_number + self.step < self.stop):
                self.seq.go2next_frame(self.step)
                return self.seq
            else:
                raise StopIteration
        except StopIteration:
            raise
        except Exception as e:
            log.error(e)
            raise StopIteration

# eof
