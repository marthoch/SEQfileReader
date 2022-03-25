#!/usr/bin/env python3
__author__ = 'Martin Hochwallner <marthoch@users.noreply.github.com>'
__email__ = "marthoch@users.noreply.github.com"
__license__ = "BSD 3-clause"

import numpy as np
import datetime
import pandas as pd

try:
    import fnv
    import fnv.reduce
    import fnv.file
    # documentation file:///C:/Program%20Files/FLIR%20Systems/sdks/file/python/doc/index.html
except Exception as e:
    print("""
download "FLIR Science File SDK" from https://flir.custhelp.com/app/account/fl_download_software
install Python module according https://flir.custhelp.com/app/answers/detail/a_id/3504/~/getting-started-with-flir-science-file-sdk-for-python
    Use the setup.py attached to the installation help.
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
        self.filename = filename
        self.im = fnv.file.ImagerFile(self.filename)
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

    def __str__(self):
        return """SEQfileReader(filename=r'{s.filename}')
    unit: ............... {s.im.unit.name} / {s.im.temp_type.name} 
    image size: ......... {s.im.width} x {s.im.height}
    number of frames: ... {s.im.num_frames}
    recording start time: {recStartTime}
    recording time: ..... {recordingTime} sec
    frame rate: ......... {s.frameRate} Hz
    current frame:
        frame number: ....... {s.im.frame_number}
        preset number: ...... {s.im.num_presets}
        date/time of frame:   {frame_info_time}
        """.format(s=self,
                   frame_info_time=self.im.frame_info.time.isoformat(),
                   recStartTime=self._firstFrame_time.isoformat(),
                   recordingTime=self.recordingTime.total_seconds())

    __repr__ = __str__

    def get_image(self):
        return np.array(self.im.final, copy=False).reshape((self.im.height, self.im.width))

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
            for fr in self.im.frame_iter(0):
                t[fr.frame_number] = fr.frame_info.time
            self.timeArray = pd.DataFrame({'time_file': t})
        return self.timeArray

    def read_full_numpy(self, selection=None, frame_range=None):
        """
        selection=[x1,y1, x2,y2]
        frameRange=[startFrame, endFrame]
        """
        if frame_range is None:
            n_frames = self.im.num_frames
            frame_range = [0, self.im.num_frames]
        else:
            n_frames = frame_range[1] - frame_range[0]
        if selection is None:
            h = self.im.height
            w = self.im.width
        else:
            h = selection[3] - selection[1]
            w = selection[2] - selection[0]

        # preallocate arrays
        fi = np.empty(n_frames, dtype=np.uint32)
        t = np.empty(n_frames, dtype=datetime.datetime)
        v = np.empty([h, w, n_frames], dtype=np.float32)

        # for fr in self.im.frame_iter(0):
        for i, fn in enumerate(range(*frame_range)):
            self.im.get_frame(fn)
            fr = self.im
            fi[i] = fr.frame_number
            t[i] = fr.frame_info.time
            img = np.array(fr.final, copy=False).reshape((fr.height, fr.width))
            if selection is None:
                v[:, :, i] = img
            else:
                v[:, :, i] = img[selection[1]:selection[3], selection[0]:selection[2]]
        if frame_range is None:  # when reading all
            self.timeArray = pd.DataFrame({'time': t})
        return t, v, fi

# eof
