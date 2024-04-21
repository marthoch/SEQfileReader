#!/usr/bin/env python3
__author__  = 'Martin Hochwallner <marthoch@users.noreply.github.com>'
__email__   = "marthoch@users.noreply.github.com"
__license__ = "BSD 3-clause"

import os
import numpy as np
import scipy as sp
import datetime
import pandas as pd
from numpy import ndarray
import matplotlib.pyplot as plt

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
    """SEQfileReader: High level reader for FLIR *.seq, and *.csq files (thermography recordings (IR))
    based on the FLIR Science File SDK

>>> seq = SEQfileReader(filename=r'path to file.seq', in_celsius=True)

>>> seq
SEQfile(filename=r'testRecording.seq')
    camera: ............. FLIR A655sc / 55001-0302 / SN:0
    lens: ............... FOL25 / T197909 / SN:0
    interlaced: ......... False
    unit: ............... TEMPERATURE_FACTORY / CELSIUS
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

    frameRate_nominal_available = np.concatenate([200. / 2 ** np.arange(0, 5), # A655sc
                                                  30. / 2 ** np.arange(0, 3)]) # T650sc # FIXME: ???

    def __init__(self, filename, in_celsius=False, frameRate_forced=None):
        """

:param filename:

:param in_celsius:
    False: temperatur in Kelvin
    True:  temperture in Celcius

:param frameRate_forced:
    None: frame rate is estimated
    frame rate in Hz: value is forced
        """
        log.debug('opening "%s"', filename)
        self._filename = filename
        self.im = fnv.file.ImagerFile(self._filename)
        self.im.unit = fnv.Unit.TEMPERATURE_FACTORY
        if in_celsius:
            self.im.temp_type = fnv.TempType.CELSIUS
        else:
            self.im.temp_type = fnv.TempType.KELVIN

        self.camera             = self.im.source_info.camera  # .camera_model
        self.camera_part_number = self.im.source_info.camera_part_number
        self.camera_serial      = self.im.source_info.camera_serial
        self.lens             = self.im.source_info.lens
        self.lens_part_number = self.im.source_info.lens_part_number
        self.lens_serial      = self.im.source_info.lens_serial

        self.interlaced = self.im.source_info.interlaced
        self.source_time = self.im.source_info.time

        self.go2frame(0)
        self._firstFrame_time = self.im.frame_info.time
        self.go2frame(-1)
        self._lastFrame_time = self.im.frame_info.time
        self.timeArray = None
        if frameRate_forced is None:
            self.frameRate_forced = False
            self._estimate_framerate()
        else:
            self.frameRate_forced = True
            self.frameRate = frameRate_forced
        self.recordingTime = self._lastFrame_time - self._firstFrame_time + datetime.timedelta(seconds=1/self.frameRate)

        # reset
        self.go2frame(0)


    def _estimate_framerate(self):
        # old: fr_ = 1 / ((self._lastFrame_time - self._firstFrame_time).total_seconds() / (self.im.num_frames - 1))
        timedf = self.read_time_df()
        tdiff = timedf.timestamp.diff().dt.total_seconds()
        tdiff_median = tdiff.median()
        tdiff_err = np.abs(tdiff - tdiff_median).max()
        if tdiff_err / tdiff_median > 0.1:
            log.warning(f'The time deviates up to {tdiff_err:.3} s or {tdiff_err / tdiff_median:.3} x sample time' )
        fr_ = 1/tdiff_median
        idx = (np.abs(self.frameRate_nominal_available - fr_)).argmin()
        self.frameRate = self.frameRate_nominal_available[idx]
        if np.abs(self.frameRate - fr_)/fr_ > 0.01:
            log.warning(f'Estimated frame rate, {fr_:.3} Hz, diverges from predefined frame rate {self.frameRate:.3}')


    def analyse_framerate(self):
        timedf = self.read_time_df()
        tdiff = timedf.timestamp.diff().dt.total_seconds()

        T = self.get_time()
        T_err = (T.timestamp - T.time_nom).dt.total_seconds().abs().max()

        print(f"""Relative frame rate deviation:
{tdiff.describe()}

Absolute frame rate deviation:
{((tdiff - 1/self.frameRate).describe())}

Max error between time in file and nominal time: {T_err} s
""")
        fig, (ax, ax2) = plt.subplots(2,1)
        r_tdiff = tdiff * self.frameRate
        r_tdiff.hist(ax=ax, bins = np.arange(-0.5, int(r_tdiff.max()+2)))
        ax.set_xlabel('multiples of frame rate')
        ax.grid(True)

        ax2.plot(timedf.timestamp, (tdiff * self.frameRate - 1))
        ax2.axhline(0.5, color='red', lw=1, zorder=-1)
        ax2.set_xlabel('time')
        ax2.set_ylabel('nr dropped frames')
        ax2.grid(True)

        return fig


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
        return f"""SEQfileReader(filename=r'{self.filename}')
    camera: ............. {self.camera} / {self.camera_part_number} / SN:{self.camera_serial}
    lens: ............... {self.lens} / {self.lens_part_number} / SN:{self.lens_serial}
    interlaced: ......... {self.interlaced}
    unit: ............... {self.im.unit.name} / {self.im.temp_type.name} 
    image size: ......... {self.im.width} x {self.im.height}
    number of frames: ... {self.number_of_frames}
    source time:          {self.source_time.isoformat()}
    recording start time: {self._firstFrame_time.isoformat()}
    recording time: ..... {self.recordingTime.total_seconds()} sec
    frame rate: ......... {self.frameRate} Hz {'(forced)' if self.frameRate_forced else ''}
    current frame:
        frame number: ....... {self.current_frame_number}
        preset: ............. {self.im.preset}
        date/time of frame:   {self.im.frame_info.time.isoformat()}
        emissivity:               {self.emissivity}
        atmospheric_transmission: {self.atmospheric_transmission}
        distance: ................{self.im.object_parameters.distance}
        preset: ............. {self.im.preset}  {self.im.source_info.preset_info[self.im.preset].available}
            frame rate: ..... {self.im.source_info.preset_info[self.im.preset].frame_rate} Hz (valid={self.im.source_info.preset_info[self.im.preset].frame_rate_valid}) 
            calibration range: {sp.constants.convert_temperature(self.im.source_info.preset_info[self.im.preset].min_temp, 'K', 'C')} - {sp.constants.convert_temperature(self.im.source_info.preset_info[self.im.preset].max_temp, 'K', 'C')} degC
"""

    __repr__ = __str__


    def get_frame(self):
        return np.array(self.im.final, copy=False).reshape((self.im.height, self.im.width))

    get_image = get_frame  # backwards compatibility, shall not be used


    def set_active_frame(self, frame_number):
        #return self.im.get_frame(frame_number)
        raise('use go2frame instead')

    def get_time(self):
        return pd.concat([self.timeArray, self.get_time_df_nom(), self.get_time_sec0_df_nom()], axis=1)


    def get_time_df_nom(self):
        """Time vector nominally spaced (based on frame rate) as pandas DataFrame."""
        return pd.DataFrame({'time_nom': self._firstFrame_time + datetime.timedelta(days=0, seconds=(
                1 / self.frameRate)) * np.arange(self.number_of_frames)})


    def get_time_sec0_df_nom(self):
        """Time vector in sec starting with 0, nominally spaced (based on frame rate) as pandas DataFrame."""
        return pd.DataFrame({'timeSec0_nom': (1 / self.frameRate) * np.arange(self.number_of_frames)})


    def read_time_df(self, reread=False):
        if reread or (self.timeArray is None):
            t = np.empty(self.number_of_frames, dtype=datetime.datetime)
            fi: ndarray = np.empty(self.number_of_frames, dtype=np.uint32)
            for i, fr in enumerate(self.frame_iter()):
                t[i] = fr.current_frame_time
                fi[i] = fr.current_frame_number
            self.timeArray = pd.DataFrame(dict(frame_number=fi, timestamp=t))
        return self.timeArray


    def read_as_numpy(self, selection=None, start=0, stop=None, step=1):
        """Read as 3d numpy array:

        :param selection:   [[row0, row1],[col0, col1]] ... [[row from to},[col from to]]
        :param start:
        :param stop:
        :param step:

        The return format corresponds with scikit-image: (pln, row, col)
        https://scikit-image.org/docs/stable/user_guide/numpy_images.html#notes-on-the-order-of-array-dimensions
        """
        if stop is None:
            stop = self.number_of_frames
        n_frames = (stop - start -1) // step + 1
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
        for i, fr in enumerate(self.frame_iter(start=start, stop=stop, step=step)):
            fi[i] = fr.current_frame_number
            img = self.get_frame()
            if selection is None:
                v[i, :, :] = img
            else:
                v[i, :, :] = img[selection[0][0]:selection[0][1], selection[1][0]:selection[1][1]]
            T = pd.merge(self.get_time(), pd.DataFrame(dict(frame_number=fi)), on='frame_number')
        return v, T

    def plot_show_line(self, line):
        """
        line = dict(p0=dict(v=1, h=1), p1=dict(v=10, h=50), len=None)
        """
        img = self.get_frame()
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, interpolation='nearest', cmap='plasma')
        ax.plot([line['p0']['h'], ], [line['p0']['v'], ], color='lime', lw=2, alpha=0.5, marker='o',
                markersize=20)
        ax.plot([line['p1']['h'], ], [line['p1']['v'], ], color='deepskyblue', lw=2, alpha=0.5,
                marker='s', markersize=20)
        ax.plot([line['p0']['h'], line['p1']['h']], [line['p0']['v'], line['p1']['v']],
                color='lime', lw=1, alpha=0.8)
        return fig


    def read_line(self, line=None, hline=None, vline=None, interpolation='linear', return_as_df=True):
        """
        line = dict(p0=dict(v=1, h=1), p1=dict(v=10, h=50), len=None)
        interpolation='linear' or 'nearest'

        use self.plot_show_line(line) to show where the line is set.
        """
        img = self.get_frame()
        ret = helper.read_line_from_frame(img, line=line, hline=hline, vline=vline, interpolation=interpolation)
        if return_as_df:
            return pd.DataFrame.from_dict(ret)
        else:
            return ret


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


    def read_line_over_time_df(self, start=0, stop=None, step=1, line=None, hline=None, vline=None,
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
        # FIXME: stop does not work: time

        lines_over_time = self.read_line_over_time(start=start, stop=stop, step=step, line=line, hline=hline, vline=vline,
                            interpolation=interpolation)
        frame_number = self.get_time().frame_number
        T = pd.Series(lines_over_time['time'], name='time')
        T_sec = (T - T.iloc[0]).dt.total_seconds()
        T_secN = self.get_time_sec0_df_nom()['timeSec0_nom'].iloc[start:stop:step]
        T_secN.rename('time_secN')
        T_err = (T_sec - T_secN).abs().max()
        if (T_err > 0.000):
            log.warning(f'time_sec - time_secN  = {T_err}')
        pixels = pd.DataFrame(lines_over_time['values'].T)
        #log.debug(f'pixels {pixels.shape}')
        time = pd.concat([T, T_sec, T_secN, frame_number], axis=1, keys=['timestamp', 'time_sec', 'time_secN', 'frame_number'])

        #log.debug(f'time {time.shape}')
        df = pd.concat([time, pixels], axis=1, keys=['time', 'pixels'])
        # recal
        df['recalibration'] = (df.pixels.diff() == 0.0).all(axis=1)
        r = df['recalibration'].diff().cumsum() + 1
        r[r.isna()] = 1
        r[df['recalibration']] *= -1
        df['recalibration_occassion'] = r
        if df['recalibration'].any():
            log.warning('recording contains recalibration')
        return df


    def read_line_over_time_df_fixdropedframes(self, start=0, stop=None, step=1, line=None, hline=None, vline=None,
                            interpolation='linear', df=None):
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
        # FIXME: stop does not work: time
        if df is None:
            df = self.read_line_over_time_df(start=start, stop=stop, step=step, line=line, hline=hline, vline=vline,
                                             interpolation=interpolation)
        df = df.copy()
        df['interpolated'] = False
        df['k'] = (df.time.time_sec * self.frameRate).round()
        df = df.reset_index(drop=True).drop_duplicates(subset=[('k', '')]).set_index('k')
        df = df.reindex(np.arange(df.index[-1] + 1))
        df.loc[:, ('time', 'time_secN')] = (1 / self.frameRate) * np.arange(df.shape[0])
        # df.loc[:, ('time', 'timestamp')].interpolate(inplace=True)
        df.loc[:, ('time', 'time_sec')].interpolate(inplace=True)
        # df.loc[:, ('time', 'frame_number')].interpolate(inplace=True)
        for i in df.pixels.columns:
            df.loc[:, ('pixels', i)].interpolate(inplace=True)
        df.loc[:, ('interpolated', '')].fillna(True, inplace=True)
        return df


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
