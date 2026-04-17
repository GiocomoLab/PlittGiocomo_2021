"""
Batch processing pipeline for CA1 morphing sessions.

Runs the full per-session processing pipeline (load -> align -> place cells)
in parallel using ProcessPoolExecutor and saves each result as a pickle file.
This was a preprocessing step for creating NWB files.

Typical entry point::

    python -m morph_analyses.batch_sess_pkl

or import and call run_sessions() with a metadata dict (e.g.
m.mouse_metadata.rare_sessions). Completed sessions are skipped if the output
pickle already exists and has non-empty place_cell_info. Errors are caught,
logged to batch_errors.log, and do not stop the remaining sessions.
"""

import os
import traceback
import logging
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import morph_analyses as m


twop_basedir = pathlib.Path('/mnt/BigDisk/2P_scratch/TwoTower')
vr_basedir = pathlib.Path('/mnt/BigDisk/morph_vr_data/')
out_basedir = pathlib.Path('/home/mplitt/morph_sess_pkls/')

logging.basicConfig(
    filename='/home/mplitt/morph_sess_pkls/batch_errors.log',
    level=logging.ERROR,
    format='%(asctime)s %(message)s',
)

def make_f_dict(mouse, date, sess, scan, scene):
    '''Build the file-path dictionary required by CA1MorphSession.

    Constructs absolute paths for the scan binary (.sbx), scan header (.mat),
    VR behavioral data (.sqlite), and Suite2P output folder from the session
    metadata fields. All paths are returned as strings for compatibility with
    the TwoPUtils Session constructor.

    inputs: mouse - mouse identifier string (must match directory name under twop_basedir)
            date  - date string in DD_MM_YYYY format
            sess  - session number (integer)
            scan  - scan number (integer)
            scene - VR scene / track name (e.g. 'TwoTower_foraging')
    outputs: f - dict with keys 'mouse', 'scan_file', 'scanheader_file',
                 'vr_filename', 's2p_path', 'prompt_for_keys', 'VR_only', 'scanner'
    '''
    f = {
        'mouse': mouse,
        'scan_file': twop_basedir / mouse / date / scene / f"{scene}_{sess:03d}_{scan:03d}.sbx",
        'scanheader_file': twop_basedir / mouse / date / scene / f"{scene}_{sess:03d}_{scan:03d}.mat",
        'vr_filename': vr_basedir / mouse / date / f"{scene}_{sess}.sqlite",
        's2p_path': twop_basedir / mouse / date / scene / f"{scene}_{sess:03d}_{scan:03d}" / "suite2p",
        'prompt_for_keys': False,
        'VR_only': False,
        'scanner': 'NLW',
    }
    f['scan_file'] = str(f['scan_file'])
    f['scanheader_file'] = str(f['scanheader_file'])
    f['vr_filename'] = str(f['vr_filename'])
    f['s2p_path'] = str(f['s2p_path'])
    return f


def run_session(mouse, sess_deets):
    '''Run the full processing pipeline for a single session and return the result.

    Constructs file paths, initializes a CA1MorphSession, loads scan metadata,
    aligns VR to 2P, extracts trial info, loads Suite2P outputs, builds standard
    timeseries / trial matrices, and computes place cells with 1000 permutations.

    inputs: mouse      - mouse identifier string
            sess_deets - session metadata dict with keys 'date_str', 'session',
                         'scan', and 'scene'
    outputs: sess - fully processed CA1MorphSession object
    '''
    f = make_f_dict(mouse, sess_deets['date_str'], sess_deets['session'], sess_deets['scan'], sess_deets['scene'])

    sess = m.sess.CA1MorphSession(**f)
    sess.load_scan_info()
    sess.align_VR_to_2P()
    sess.get_trial_info()
    sess.load_suite2p_data()
    sess.gen_standard_ts_tmats()
    sess.place_cells_calc(nperms=1000)
    
    return sess

    
    
def _process_one(mouse, sess_deets):
    '''Process one session, saving the result to disk. Skips if already complete.

    Checks whether the output pickle already exists and has non-empty
    place_cell_info. If so, skips silently. Otherwise calls run_session and
    saves the result. Exceptions are caught, printed, and logged to
    batch_errors.log without re-raising so remaining sessions can continue.

    inputs: mouse      - mouse identifier string
            sess_deets - session metadata dict (see run_session)
    '''
    outfile = out_basedir / mouse / sess_deets['date_str'] / f"{sess_deets['scene']}_{sess_deets['session']}.pkl"
    if os.path.exists(outfile):
        sess = m.sess.CA1MorphSession.load(outfile)
        if len(list(sess.place_cell_info.keys())) > 0:
            print(f"File {outfile} already exists, skipping...")
            return
    try:
        sess = run_session(mouse, sess_deets)
        sess.save(outfile)
    except Exception as e:
        msg = f"FAILED | mouse={mouse} | session={sess_deets['session']} | date={sess_deets['date_str']} | {e}\n{traceback.format_exc()}"
        print(msg)
        logging.error(msg)


def run_sessions(sess_dict, n_workers=8):
    '''Process all sessions in sess_dict in parallel using a process pool.

    Flattens the nested metadata dict into a list of (mouse, sess_deets) tasks
    covering both training and test sessions, then dispatches them to
    _process_one via ProcessPoolExecutor. Progress and errors are printed as
    futures complete. Additional errors raised during future collection are also
    caught and logged.

    inputs: sess_dict - metadata dict keyed by mouse identifier, each value
                        containing 'test_sessions' and 'training_sessions' lists
                        (e.g. m.mouse_metadata.rare_sessions)
            n_workers - number of parallel worker processes (default 8)
    '''
    tasks = [
        (mouse, sess_deets)
        for mouse, metadata in sess_dict.items()
        for sess_deets in metadata['test_sessions'] + metadata['training_sessions']
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one, mouse, sess_deets): (mouse, sess_deets)
                   for mouse, sess_deets in tasks}
        for future in as_completed(futures):
            mouse, sess_deets = futures[future]
            try:
                future.result()
            except Exception as e:
                msg = f"FAILED | mouse={mouse} | session={sess_deets['session']} | date={sess_deets['date_str']} | {e}\n{traceback.format_exc()}"
                print(msg)
                logging.error(msg)


        
if __name__ == "__main__":
    # run_sessions(m.mouse_metadata.rare_sessions)
    # run_sessions(m.mouse_metadata.frequent_sessions)
    run_sessions(m.mouse_metadata.frequent_w_decision_sessions)