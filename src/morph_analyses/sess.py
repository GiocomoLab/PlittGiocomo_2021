import pathlib
import pickle
import numpy as np
import two_photon_utils as tpu
import morph_analyses.PlaceCellAnalysis as pc
import morph_analyses.utilities as u

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



    def add_pos_binned_trial_matrix(self, ts_name, pos_key='pos', min_pos=0, max_pos=450, bin_size=10, mat_only=True, 
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

        super(CA1MorphSession, self).add_pos_binned_trial_matrix(ts_name, pos_key,
                                                              min_pos=min_pos,
                                                              max_pos=max_pos,
                                                              bin_size=bin_size,
                                                              mat_only=mat_only,
                                                              **trial_matrix_kwargs)

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2

    def place_cells_calc(self,  **pc_kwargs):

        masks, SI, pvals = pc.place_cells_calc(self.timeseries['spks_norm'].T, 
                                            self.vr_data['pos'],
                                            self.trial_info,
                                            self.trial_start_inds, 
                                            self.teleport_inds,
                                            morphlist=np.unique(self.trial_info['morphs']).tolist(),
                                            **pc_kwargs)
        d = self.place_cell_info
        d['masks'] = masks
        d['SI'] = SI
        d['p'] = pvals


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
        
    @classmethod
    def from_nwb(cls, nwb_path, **kwargs):
        """Create a CA1MorphSession from an NWB file.

        Parameters
        ----------
        nwb_path : str or pathlib.Path
            Path to NWB file containing CA1 morphology session data.

        Returns
        -------
        CA1MorphSession
        """
        
        raise NotImplementedError("Loading from NWB not yet implemented.")
    
        inst = cls(**kwargs) # construct a minimal YMazeSession class instance

        with NWBHDF5IO(filepath, 'r') as io:
        
            nwb = io.read()

            # load metadata annotation
        
            ann = nwb.acquisition.get('trial_cell_data')
            
            meta_json = ann.data[0]
            meta = json.loads(meta_json)
            for k, v in meta.items():
                setattr(inst, k, v)
                
            if inst.mux:
                inst._load_nwb_data_sparse(nwb)
            else:
                inst._load_nwb_data_dense(nwb)

            
        return inst
    
        

