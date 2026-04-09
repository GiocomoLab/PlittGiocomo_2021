import pathlib
import pickle
import numpy as np
import two_photon_utils as tpu
import morph_analyses.place_cell_analysis as pc
import morph_analyses.utilities as u
import json
import pandas as pd

try:
    from pynwb import NWBFile, NWBHDF5IO, TimeSeries
    from pynwb.misc import AnnotationSeries
except Exception:
    NWBFile = None
    NWBHDF5IO = None
    AnnotationSeries = None
    TimeSeries = None

_TRIAL_COLS = ('morph', 'towerJitter', 'wallJitter', 'bckgndJitter')


class CA1MorphSession(tpu.sess.Session):
    """Session subclass for CA1 morphology experiment (Plitt & Giocomo 2019)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trial_info = {}
        self.place_cell_info = {}

    def get_trial_info(self):
        """Store per-trial morph and jitter values from vr_data.

        Reads morph, towerJitter, wallJitter, and bckgndJitter at each trial
        start index. Results are stored in self.trial_info as 1-D numpy arrays
        of length n_trials.
        """
        self.trial_info, _, _= u.by_trial_info(self.vr_data)
        self.trial_info['effective_morph'] = self.trial_info['morphs'] + self.trial_info['wallJitter']



    def add_pos_binned_trial_matrix(self, ts_name, pos_key='position', min_pos=0, max_pos=450, bin_size=10, mat_only=True, 
                                    **trial_matrix_kwargs):
        """

        :param ts_name:
        :param pos_key:
        :param min_pos:
        :param max_pos:
        :param bin_size:
        :param mat_only:
        :param trial_matrix_kwargs:
        :return:
        """

        super(CA1MorphSession, self).add_pos_binned_trial_matrix(
            ts_name, pos_key,
            min_pos=min_pos, max_pos=max_pos, bin_size=bin_size, mat_only=mat_only,
            **trial_matrix_kwargs
            )

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2

    def place_cells_calc(self,  **pc_kwargs):

        masks, SI, pvals = pc.place_cells_calc(self.timeseries['spks_norm'].T, 
                                            self.vr_data['position'],
                                            self.trial_info,
                                            self.trial_start_inds, 
                                            self.teleport_inds,
                                            morphlist=np.unique(self.trial_info['morphs']).tolist(),
                                            **pc_kwargs)
        d = self.place_cell_info
        d['masks'] = masks
        d['SI'] = SI
        d['p'] = pvals
        
        
    def gen_standard_ts_tmats(self):
        self.s2p_path = pathlib.Path(self.s2p_path)
        
        iscell = np.load(self.s2p_path / 'plane0' / 'iscell.npy')[:, 0].astype(bool)

        ts = {
            'F': self.s2p_path / 'plane0' / 'F.npy',
            'Fneu': self.s2p_path / 'plane0' / 'Fneu.npy',
            'spks': self.s2p_path / 'plane0' / 'spks.npy',
        }

        for key, path in ts.items():
            if path.exists():
                ts[key] = np.load(path)[iscell, :]
            else:
                print(f"Warning: {key} not found at {path}. Skipping.")
                ts[key] = None
        
        ts['dff'] = tpu.utilities.dff(ts['F'] - .7*ts['Fneu']) 
        self.add_timeseries(F=ts['F'], Fneu=ts['Fneu'], spks=ts['spks'], dff=ts['dff'])
        self.add_timeseries(spks_norm=ts['spks']/np.nanpercentile(ts['spks'], 99, axis=1, keepdims=True))

        self.add_pos_binned_trial_matrix('spks_norm')
        
        # behavior
        self.add_timeseries(
            licks=self.vr_data['lick'].to_numpy().astype(float),
            speed=self.vr_data['speed'].to_numpy().astype(float),
        )
        self.add_pos_binned_trial_matrix('speed')
        self.add_pos_binned_trial_matrix('licks')
        


    def save(self, path=None):
        """Serialize this session to a pickle file.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination file path. If None, saves to
            ``<basedir_2P>/<mouse>/<date>/session.pkl`` using session attributes.

        Returns
        -------
        pathlib.Path
            The path the file was written to.
        """
        if path is None:
            path = pathlib.Path(self.s2p_path) / 'session.pkl'
        else:
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path):
        """Load a pickled CA1MorphSession.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the pickle file written by :meth:`save`.

        Returns
        -------
        CA1MorphSession
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_nwb(cls, mouse, traintest, day, **kwargs):
        """Create a CA1MorphSession from an NWB file.

        Parameters
        ----------
        
        Returns
        -------
        CA1MorphSession
        """
        from . import params
        assert traintest in set(('training','testing')), "traintest must be either 'training' or 'testing'"

        day = str(day).replace('.','-')
        nwb_path = params.dandi_dir / f'sub-{mouse}' / f'sub-{mouse}_ses-{traintest}-day{day}_behavior+ophys.nwb'

        
        inst = cls(**kwargs) # construct a minimal YMazeSession class instance

        with NWBHDF5IO(nwb_path, 'r') as io:
        
            nwb = io.read()

            # load metadata annotation
        
            ann = nwb.acquisition.get('trial_data')
            
            meta_json = ann.data[0]
            meta = json.loads(meta_json)
            for k, v in meta.items():
                setattr(inst, k, v)
                
            
            inst._load_nwb_data(nwb)
            
        return inst
    
    def _load_nwb_data(self, nwb):
        import copy 
        self.trial_start_inds = np.array(self.trial_start_inds)
        self.teleport_inds = np.array(self.teleport_inds)

        for k, v in self.trial_info.items():
            self.trial_info[k]=np.array(v)

        self._place_cell_info = {}
        for k, pc_dict in self.place_cell_info.items():
            self._place_cell_info[k] = {}
            for _k, v in pc_dict.items():
                self.place_cell_info[k][_k] = np.array(v)
                self._place_cell_info[k][float(_k)] = np.array(v)
        self.place_cell_info = self._place_cell_info
            

        # load VR timeseries
        behav_module = nwb.processing['behavior']['2P-aligned behavior']
        vr_data = {k:pd.Series(v.data[:]) for k,v in behav_module.time_series.items()}
        
        vr_data['time'] = pd.Series(behav_module['position'].timestamps[:])
        
        self.vr_data = pd.DataFrame(vr_data)

        # load 2P data
        ophys_module = nwb.processing['ophys']

        timeseries = {
            'F': ophys_module['fluorescence']['fluorescence'].data[:].T,
            'Fneu': ophys_module['neuropil']['neuropil fluorescence'].data[:].T,
            'dff': ophys_module['dF']['dF'].data[:].T,
            'spks': ophys_module['deconvolved activity']['deconvolved activity'].data[:].T,
            'spks_norm': ophys_module['normalized deconvolved activity']['normalized deconvolved activity'].data[:].T,
            'licks': vr_data['licks'].to_numpy()[np.newaxis,:],
            'reward': vr_data['reward'].to_numpy()[np.newaxis,:],
            'speed': vr_data['speed'].to_numpy()[np.newaxis,:],
            }

        self.add_timeseries(**timeseries)
        self.add_pos_binned_trial_matrix(list(timeseries.keys()), pos_key='position')

        meanImg = ophys_module['Backgrounds']['meanImg'].data[:]
        self.s2p_ops = {
            'meanImg': meanImg,
            'Ly': meanImg.shape[0],
            'Lx': meanImg.shape[1],
            'Fs': ophys_module['fluorescence']['fluorescence'].rate,
            }
        self.frame_rate = ophys_module['fluorescence']['fluorescence'].rate
        
        self.s2p_stats = []
        pixel_mask_table = ophys_module['ImageSegmentation']['PlaneSegmentation'].to_dataframe()
        for ind, row in pixel_mask_table.iterrows():
            roi_dict = {'ypix': [], 'xpix': [], 'lam': []}
            for px in row['pixel_mask']:
                roi_dict['ypix'].append(px[0])
                roi_dict['xpix'].append(px[1])
                roi_dict['lam'].append(px[2])
            for k,v in roi_dict.items():
                roi_dict[k] = np.array(v)
            self.s2p_stats.append(roi_dict)
        self.s2p_stats = np.array(self.s2p_stats)

