import os
import traceback
import logging
import pathlib
import pickletools
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