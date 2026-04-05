import datetime
import copy
import pathlib
import subprocess
from uuid import uuid4
import hdmf
from hdmf.backends.hdf5.h5_utils import H5DataIO
import json

import numpy as np
import scipy
from natsort import natsorted
import pynwb
from pynwb.file import Subject


import suite2p
from pynwb import NWBHDF5IO, NWBFile
from pynwb.misc import AnnotationSeries
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
    DfOverF,
)

from pynwb.behavior import BehavioralTimeSeries

import STX3KO_analyses as stx
import TwoPUtils as tpu



# SCRATCH_DIR = pathlib.Path("/mnt/BigDisk/2P_scratch")
# VR_DIR = pathlib.Path("/mnt/BigDisk/VRData")


OUTPATH = pathlib.Path("/mnt/BigDisk/NWB_files")
SESSPATH = pathlib.Path("/home/mplitt/YMazeSessPkls")
VRSESSPATH = pathlib.Path("/home/mplitt/YMaze_VR_Pkls")
SBXMATPATH = pathlib.Path("/mnt/BigDisk/2P_scratch")


class SessNWBConverter:
    
    def __init__(self, mouse, metadata, session, day, oak_pwd, scan=0, sub_notes=''):

        self.mouse = mouse
        if mouse in stx.mouse_metadata.ctrl_sessions.keys():
            self.sub_description = f"Control mouse. Viruses: {metadata.get('functional_indicator')} \
                {metadata.get('static_indicator')}"
        elif mouse in stx.mouse_metadata.cre_sessions.keys():
            self.sub_description = f"Cre mouse. Viruses: {metadata.get('functional_indicator')} \
                {metadata.get('static_indicator')}"
        else:
            raise ValueError("Mouse name must be in ctrl or cre mice metadata")
        
        self.session = session
        self.metadata = metadata
        self._oak_pwd = oak_pwd
        self.day = day
        self.scan = scan
        self.sub_notes = sub_notes

        self.sess_path = SESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl" 
        self.sess = stx.session.YMazeSession.from_file(self.sess_path, novel_arm = session.get('novel_arm'))
        
        self.vr_sess_path = VRSESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl"
        self.vr_sess = stx.session.YMazeSession.from_file(self.vr_sess_path, novel_arm = session.get('novel_arm'))
        
        self.sbx_mat_path = SBXMATPATH / mouse / session.get('date_str') / \
            f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.mat"
        # self.sbx_path = SBXMATPATH / mouse / session.get('date_str') / \
        #     f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.sbx"
        self.sbx_mat_path.parent.mkdir(exist_ok=True, parents=True)
        self.sbx_mat = None
        
        self.nwb_file = None
        self.behav_module = None
        self.ophys_module = None
        self.roi_table = None
        

        self.out_path = OUTPATH / mouse / f"ymaze_day{day}_scan{scan}_ophys_behav.nwb"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _get_ttl_times(self):
        if self.sbx_mat is None:
            self._load_sbx_mat()
            
        

        fr = self.sbx_mat['frame_rate'] # frame rate
        lr = fr * self.sbx_mat['config']['lines']/self.sbx_mat['fov_repeats']  # line rate

        frames = self.sbx_mat['frame'].astype(int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        
        frames = frames * self.sbx_mat['fov_repeats']
        if self.sbx_mat['fold_lines']>0:
            lines = np.array([l % self.sbx_mat['fold_lines'] for l in self.sbx_mat['line']])
        else:
            lines = np.array(self.sbx_mat['line'])

        ttl_times = frames / fr + lines / lr
        return ttl_times

    def init_nwb_file(self):
        self.nwb_file = NWBFile(
            session_description = "Preprocessed 2P and VR Data",
            session_start_time = datetime.datetime.now().astimezone(),
            identifier=str(uuid4()),  # required
            experimenter = ['Plitt, Mark'],
            lab="Lisa Giocomo",
            institution="Stanford University",
            session_id=f"ymaze_day{self.day}_scan{self.scan}_novel_arm{self.session.get('novel_arm')}",
            experiment_description =  f"YMaze day {self.day}. Novel arm = {self.session.get('novel_arm')}." + self.sub_notes,
            related_publications='https://doi.org/10.1101/2023.11.20.567978 ',
            keywords=["two photon", "hipppocampus", "CA1", "syntaxin3"]
        )

        self.nwb_file.subject = Subject(
            subject_id = self.metadata.get('alias'),
            age = self.session.get('datetime') - self.metadata.get('date_of_birth'),
            description = self.sub_description,
            species = 'Mus musculus',
            sex = self.metadata.get('sex'),
            genotype = self.session.get('genotype'),
        )
        
        self.behavior_module = self.nwb_file.create_processing_module('behavior', 'VR behavioral timeseries')


    def add_vr_data_full_res(self):
        ts_cntnr = BehavioralTimeSeries(name = 'Full temporal resolution behavior')
        
        time_stamps = self._get_ttl_times()
        
        
        vr_timeseries = {k: v[:,-time_stamps.shape[0]:] for k,v in self.vr_sess.timeseries.items()}
        vr_data = self.vr_sess.vr_data.iloc[-time_stamps.shape[0]:]
        
        
        # vr_data info
        
        # trial num 
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = 'current trial number',
            timestamps = time_stamps,
        )
        
        # t - spline position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['t'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                    in the grey hallway prior to trial start ",
        )
        
        
        # dz 
        ts_cntnr.create_timeseries(
            name = 'rotary encoder reading',
            data = vr_data['dz'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            description = "Scaled rotary encoder output. Raw speed of mouse. During timeouts, visual speed is 0",
        )
        
        # tstart
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial start time",
        )
        # teleport
        ts_cntnr.create_timeseries(
            name = 'trial end',
            data = vr_data['teleport'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial end/teleport time",
        )
        
        
        # manrewards
        ts_cntnr.create_timeseries(
            name = 'manual rewards',
            data = vr_data['manrewards'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Manually delivered reward, typically for solenoid failure or to unclog line",
        )
        
        # vr_timeseries 
        # speed
        ts_cntnr.create_timeseries(
            name = 'speed',
            data = vr_timeseries['speed'].ravel(),
            unit = '10 cm/s',
            timestamps = time_stamps,
            description = "Speed along Y maze",
        )
        
        
        # nonconsum_licks
        ts_cntnr.create_timeseries(
            name = 'non-consummatory licks',
            data = vr_timeseries['nonconsum_licks'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Licks outside of reward consumption. Note this may contain artifacts from periods when \n \
                there is excess liquid on the capacitive sensor",
        )
        
        ts_cntnr.create_timeseries(
            name = 'consummatory licks',
            data = vr_timeseries['consum_licks'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Licks during reward consumption",
        )
        
        # reward
        ts_cntnr.create_timeseries(
            name = 'reward',
            data = vr_timeseries['reward'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Reward delivery times.",
        )
        
        
        self.behavior_module.add(ts_cntnr)

    def add_vr_data_aligned(self):
        ts_cntnr = BehavioralTimeSeries(name = '2P-aligned behavior')
        

        
        
        vr_timeseries = self.sess.timeseries
        vr_data = self.sess.vr_data
        time_stamps = vr_data['time'].to_numpy()
        rate = self.sess.s2p_ops['fs']
        
        
        # vr_data info
        
        # trial num 
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = 'current trial number',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
        )
        
        # t - spline position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['t'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                    in the grey hallway prior to trial start ",
        )
        
        # posx 
        ts_cntnr.create_timeseries(
            name = 'x position',
            data = vr_data['posx'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Unity units x position on 2D plane",
            
        )
        
        # posz
        ts_cntnr.create_timeseries(
            name = 'y position',
            data = vr_data['posz'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Unity units y position on 2D plane",
        )
        
        # tstart
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Boolean. Trial start time",
        )
        # teleport
        ts_cntnr.create_timeseries(
            name = 'trial end',
            data = vr_data['teleport'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Boolean. Trial end/teleport time",
        )
        
        # LR
        ts_cntnr.create_timeseries(
            name = 'left or right',
            data = vr_data['LR'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "-1 = left trial. 1 = right trial",
        )
        
        # vr_timeseries 
        # speed
        ts_cntnr.create_timeseries(
            name = 'speed',
            data = vr_timeseries['speed'].ravel(),
            unit = '10 cm/s',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Speed along Y maze",
        )
        
        # block
        ts_cntnr.create_timeseries(
            name = 'block',
            data = vr_timeseries['block_number'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "current block",
        )
        
        # lick rate
        ts_cntnr.create_timeseries(
            name = 'licks',
            data = vr_data['lick'].to_numpy(),
            unit = 'avg number of licks',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Average downsampled lick rate. Do not use for quantitative lick comparisons between groups",
        )
        
        # reward
        ts_cntnr.create_timeseries(
            name = 'reward',
            data = vr_timeseries['reward'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            # rate=rate,
            # starting_time=time_stamps[0],
            description = "Reward delivery times.",
        )
        
        self.behavior_module.add(ts_cntnr)
        
    def init_2p_data(self):
        device = self.nwb_file.create_device(
            name="Microscope",
            description="Giocomo lab Neurolabware 2P Scope",
            manufacturer="Neurolabware",
        )
        
        optical_channel0 = OpticalChannel(
            name="Green PMT",
            description="an optical channel",
            emission_lambda=500.0,
        )
        

        imaging_plane = self.nwb_file.create_imaging_plane(
            name="ImagingPlane",
            optical_channel=[optical_channel0,],
            indicator='channel 0: GCaMP',
            imaging_rate=self.sess.s2p_ops["fs"],
            description="CA1 pyramidal cell layer",
            device=device,
            excitation_lambda=self.metadata.get('imaging_lambda'),
            location="CA1",
            grid_spacing=([1000/512., 1000/796.]),
            grid_spacing_unit="microns",
        )
        
        
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name="PlaneSegmentation",
            description="Suite2P output",
            imaging_plane=imaging_plane,
        )
        
        self.ophys_module = self.nwb_file.create_processing_module(
            name="ophys", description="2P imaging data"
        )
        self.ophys_module.add(img_seg)
        
        stat = self.sess.s2p_stats
        for n in range(len(stat)):
            pixel_mask = np.array(
                [stat[n]['ypix'], stat[n]['xpix'], stat[n]['lam']]
            )
            ps.add_roi(pixel_mask=pixel_mask.T)
            
        self.roi_table = ps.create_roi_table_region(
            region = [i for i in range(len(stat))],
            description="ROIs"
        )
        
        images = Images("Backgrounds", description='motion aligned average images')
        images.add_image(GrayscaleImage(name='meanImg', 
                                        data=self.sess.s2p_ops['meanImg'],
                                        description='average channel 0 (gcamp) image'))
       
        self.ophys_module.add(images)


    def add_cell_timeseries(self):
        if self.ophys_module is None:
            self.init_2p_data()
            
        F = self.sess.timeseries.get('F')
        roi_resp_series = RoiResponseSeries(
            name = 'fluorescence',
            data = F.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'raw fluorescence from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='fluorescence')
        self.ophys_module.add(fl)
        
        Fneu = self.sess.timeseries.get('Fneu')
        roi_resp_series = RoiResponseSeries(
            name = 'neuropil fluorescence',
            data = Fneu.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'raw neuropil fluorescence from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='neuropil')
        self.ophys_module.add(fl)
        
        dff = self.sess.timeseries.get('F_dff')
        roi_resp_series = RoiResponseSeries(
            name = 'dF',
            data = dff.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'dF/F from channel 0 (gcamp)',
        )
        fl = DfOverF(roi_response_series=roi_resp_series, name='dF')
        self.ophys_module.add(fl)
        
        
    def build_file(self):
        self.init_nwb_file()
        self.add_vr_data_full_res()
        self.add_vr_data_aligned()
        self.init_2p_data()
        self.add_cell_timeseries()
        self.add_trial_data()
        return self
        
    def write_file(self):
        with NWBHDF5IO(self.out_path, "w") as fio:
            fio.write(self.nwb_file)
            
    def remove_sbx_data(self):
        self.sbx_mat_path.unlink(missing_ok=True)
        # self.sbx_path.unlink(missing_ok=True)

    def _to_json_serializable(self, obj):
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(o) for o in obj]
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        try:
            import pandas as _pd

            if isinstance(obj, _pd.DataFrame):
                return obj.to_dict(orient='list')
            if isinstance(obj, _pd.Series):
                return obj.to_list()
        except Exception:
            pass
        return str(obj)

    def add_trial_data(self):
        print(self.metadata['alias'])
        meta = {
            'mouse': self._to_json_serializable(self.metadata['alias']),
            'mux': False,
            'novel_arm': self._to_json_serializable(self.sess.novel_arm),
            'day': self._to_json_serializable(self.day),
            'novel_arm': self._to_json_serializable(self.sess.novel_arm),
            'trial_info': self._to_json_serializable(getattr(self.sess, 'trial_info', None)),
            'trial_start_inds': self._to_json_serializable(self.sess.trial_start_inds.to_numpy()),
            'teleport_inds': self._to_json_serializable(self.sess.teleport_inds.to_numpy()),
            'place_cell_info': self._to_json_serializable(getattr(self.sess, 'place_cell_info', None)),
            'vr_trial_info': self._to_json_serializable(getattr(self.vr_sess, 'trial_info', None)),
            # 'vr_trial_start_inds': self._to_json_serializable(self.vr_sess.trial_start_inds.tolist()),
            # 'vr_teleport_inds': self._to_json_serializable(self.vr_sess.teleport_inds.tolist()),   
        }

        meta_json = json.dumps(meta)
        ann = AnnotationSeries(name='trial_cell_data', data=[meta_json], timestamps=[0.0])
        self.nwb_file.add_acquisition(ann)
    








                


