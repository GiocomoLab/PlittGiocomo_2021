"""
Configuration parameters for 2-photon analysis.

This module contains path configurations, DANDI dataset information,
and mouse identifier mappings used throughout the analysis pipeline.
"""

import pathlib

# DANDI dataset and directory configuration
dandiset_id = "000054"  # DANDI dataset ID for this project
dandi_download_dir = '/home/mplitt/'  # Base directory where DANDI data is/will be saved

# Repository configuration
repo_dir = '/home/mplitt/repos/PlittGiocomo_CA1Morph_2019'  # Root repository directory

# Create Path objects for key directories
dandi_download_dir = pathlib.Path(dandi_download_dir)
dandi_dir = pathlib.Path(dandi_download_dir) / dandiset_id  # Full path to DANDI dataset

repo_dir = pathlib.Path(repo_dir)
data_dir = repo_dir / 'data' 
fig_output_dir = repo_dir / 'data' / 'figures' # Default directory for saving figures
frame_grabber_dir = repo_dir / 'data' / 'FrameGrabber'

fig_output_dir.mkdir(exist_ok=True, parents=True)
frame_grabber_dir.mkdir(exist_ok=True, parents=True)
