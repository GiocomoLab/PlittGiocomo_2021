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

import morph_analyses as m



# SCRATCH_DIR = pathlib.Path("/mnt/BigDisk/2P_scratch")
# VR_DIR = pathlib.Path("/mnt/BigDisk/VRData")


OUTPATH = pathlib.Path("/home/mplitt/NWB_files")
SESSPATH = pathlib.Path("/home/mplitt/morph_sess_pkls")
SBXMATPATH = pathlib.Path("/mnt/BigDisk/2P_scratch")


class SessNWBConverter:
    
    def __init__(self, mouse, metadata, session, training_session=False, scan=0, sub_notes=''):

        self.mouse = mouse
        if mouse in m.mouse_metadata.rare_sessions.keys():
            self.sub_description = "Rare morph mouse"
        elif mouse in m.mouse_metadata.frequent_sessions.keys():
            self.sub_description = f"Frequent morph mouse"
        elif mouse in m.mouse_metadata.frequent_w_decision_sessions.keys():
            self.sub_description = f"Frequent morph with decision mouse (Extended Data Fig 10)"
        else:
            raise ValueError("Mouse name not in metadata")
        
        self.session = session
        self.metadata = metadata
        self.day = session.get('day')
        self.scan = scan
        self.sub_notes = sub_notes

        self.sess_path = SESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl" 
        self.sess = m.sess.CA1MorphSession.load(self.sess_path)
        
        # self.sbx_mat_path = SBXMATPATH / mouse / session.get('date_str') / \
        #     f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.mat"
        # # self.sbx_path = SBXMATPATH / mouse / session.get('date_str') / \
        # #     f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.sbx"
        # self.sbx_mat_path.parent.mkdir(exist_ok=True, parents=True)
        # self.sbx_mat = None
        
        self.nwb_file = None
        self.behav_module = None
        self.ophys_module = None
        self.roi_table = None
        self.training_session = training_session
        

        self.out_path = OUTPATH / mouse / f"morph_task_day{session['day']}_scan{scan}_ophys_behav.nwb"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        
    def init_nwb_file(self):
        if self.training_session:
            session_id = f"training_day{self.day}"
            description = f"Training session. Morph task. " + self.sub_notes
        else:
            session_id = f"testing_day{self.day}"
            description = f"Testing session. Morph task. " + self.sub_notes

        self.nwb_file = NWBFile(
            session_description = "Preprocessed 2P and VR Data",
            session_start_time = datetime.datetime.now().astimezone(),
            identifier=str(uuid4()),  # required
            experimenter = ['Plitt, Mark'],
            lab="Lisa Giocomo",
            institution="Stanford University",
            session_id=session_id,
            experiment_description =  description,
            related_publications='https://doi-org/10.1038/s41593-021-00816-6',
            keywords=["two photon", "hipppocampus", "CA1", "remapping"]
        )

        session_date = datetime.datetime.strptime(self.session['date_str'], '%d_%m_%Y')
        age_days = (session_date - self.metadata.get('date_of_birth')).days
        self.nwb_file.subject = Subject(
            subject_id = self.metadata.get('alias'),
            age = f"P{age_days}D",
            description = self.sub_description,
            species = 'Mus musculus',
            sex = self.metadata.get('sex'),
            genotype = self.metadata.get('genotype'),
        )
        
        self.behavior_module = self.nwb_file.create_processing_module('behavior', 'VR behavioral timeseries')


    def add_vr_data_aligned(self):
        ts_cntnr = BehavioralTimeSeries(name = '2P-aligned behavior')
        

        
        
        vr_timeseries = self.sess.timeseries
        vr_data = self.sess.vr_data
        time_stamps = vr_data['time'].to_numpy(dtype=float)
        rate = self.sess.s2p_ops['fs']
        
        
        # vr_data info
        
        # trial num
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(dtype=float),
            unit = 'arbitrary',
            description = 'current trial number',
            timestamps = time_stamps,
        )

        # pos -  position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['pos'].to_numpy(dtype=float),
            unit = '1 cm',
            timestamps = time_stamps,
            description = "Position along track. Negative values indicate inter-trial interval tunnel. 0=start of maze",
        )

        # trial start
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial start time",
        )

        # teleport
        ts_cntnr.create_timeseries(
            name = 'teleport',
            data = vr_data['teleport'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial end/teleport time",
        )

        # vr_timeseries
        # speed
        ts_cntnr.create_timeseries(
            name = 'speed',
            data = vr_timeseries['speed'].ravel().astype(float),
            unit = 'cm/s',
            timestamps = time_stamps,
            description = "Speed along Y maze",
        )


        # lick rate
        ts_cntnr.create_timeseries(
            name = 'licks',
            data = vr_data['lick'].to_numpy(dtype=float),
            unit = 'avg number of licks',
            timestamps = time_stamps,
            description = "Average downsampled lick rate. Do not use for quantitative lick comparisons between groups",
        )

        # reward
        ts_cntnr.create_timeseries(
            name = 'reward',
            data = vr_data['reward'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Reward delivery times.",
        )

        # morph value
        ts_cntnr.create_timeseries(
            name = 'morph value',
            data = vr_data['morph'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Morph value for each trial",
        )

        # wall jitter
        ts_cntnr.create_timeseries(
            name = 'wall jitter',
            data = vr_data['wallJitter'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Wall morph jitter value for each trial",
        )

        # tower jitter
        ts_cntnr.create_timeseries(
            name = 'tower jitter',
            data = vr_data['towerJitter'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Tower morph jitter value for each trial",
        )

        # background jitter
        ts_cntnr.create_timeseries(
            name = 'background jitter',
            data = vr_data['bckgndJitter'].to_numpy(dtype=float),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Background morph jitter value for each trial",
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
            excitation_lambda=float(self.metadata.get('imaging_lambda')),
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
                                        description='average gcamp image'))
       
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
        
        dff = self.sess.timeseries.get('dff')
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
        
        spks = self.sess.timeseries.get('spks')
        roi_resp_series = RoiResponseSeries(
            name = 'deconvolved activity',
            data = spks.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'deconvolved activity from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='deconvolved activity')
        self.ophys_module.add(fl)
        
        spks_norm = self.sess.timeseries.get('spks_norm')
        roi_resp_series = RoiResponseSeries(
            name = 'normalized deconvolved activity',
            data = spks_norm.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'normalized deconvolved activity from channel 0 (gcamp). Normalization is done by dividing each cell by the 99th percentile of its deconvolved activity across the session.',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='normalized deconvolved activity')
        self.ophys_module.add(fl)

        
        
    def build_file(self):
        self.init_nwb_file()
        self.add_vr_data_aligned()
        self.init_2p_data()
        self.add_cell_timeseries()
        self.add_trial_data()
        return self
        
    def write_file(self):
        with NWBHDF5IO(self.out_path, "w") as fio:
            fio.write(self.nwb_file)
            
    # def remove_sbx_data(self):
    #     self.sbx_mat_path.unlink(missing_ok=True)
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
            'day': self._to_json_serializable(self.day),
            'trial_info': self._to_json_serializable(getattr(self.sess, 'trial_info', None)),
            'trial_start_inds': self._to_json_serializable(self.sess.trial_start_inds.to_numpy()),
            'teleport_inds': self._to_json_serializable(self.sess.teleport_inds.to_numpy()),
            'place_cell_info': self._to_json_serializable(getattr(self.sess, 'place_cell_info', None)),  
        }

        meta_json = json.dumps(meta)
        ann = AnnotationSeries(name='trial_data', data=[meta_json], timestamps=[0.0])
        self.nwb_file.add_acquisition(ann)
    








                


