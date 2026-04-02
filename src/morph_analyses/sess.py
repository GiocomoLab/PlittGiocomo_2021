import numpy as np
import two_photon_utils as tpu
# from two_photon_utils.sess import Session

_TRIAL_COLS = ('morph', 'towerJitter', 'wallJitter', 'bckgndJitter')


class CA1MorphSession(tpu.sess.Session):
    """Session subclass for CA1 morphology experiment (Plitt & Giocomo 2019)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trial_info = {}

    def get_trial_info(self):
        """Store per-trial morph and jitter values from vr_data.

        Reads morph, towerJitter, wallJitter, and bckgndJitter at each trial
        start index. Results are stored in self.trial_info as 1-D numpy arrays
        of length n_trials.
        """
        n_trials = min(len(self.trial_start_inds), len(self.teleport_inds))
        starts = self.trial_start_inds[:n_trials]
        for col in _TRIAL_COLS:
            self.trial_info[col] = self.vr_data.loc[starts, col].to_numpy(dtype=float)

        self.trial_info['effective_morph'] = self.trial_info['morph'] + self.trial_info['wallJitter']

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

        # S_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(S,
                                                    # VRDat['pos']._values,tstart_inds,
                                                    # teleport_inds,bin_size=10,
                                                    # speed = VRDat['speed']._values)

        super(CA1MorphSession, self).add_pos_binned_trial_matrix(ts_name, pos_key,
                                                              min_pos=min_pos,
                                                              max_pos=max_pos,
                                                              bin_size=bin_size,
                                                              mat_only=mat_only,
                                                              **trial_matrix_kwargs)

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2

    def place_cells_calc(self, Fkey='F_dff', trial_mask=None, lr_split=True, out_key=None, min_pos=13, max_pos=43,
                         bin_size=1, mux = False, **pc_kwargs):

        masks, FR, SI = pc.place_cells_calc(S, VRDat['pos']._values,trial_info,
                        tstart_inds, teleport_inds,
                        speed=VRDat.speed._values,
                        win_trial_perm=True,morphlist=np.unique(trial_info['morphs']).tolist())



    def gen_standard_ts_tmats(self):
        pass
        

