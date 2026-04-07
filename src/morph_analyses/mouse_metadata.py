import pandas as pd
import datetime

rare_sessions = {

    '4139265.3':{
        'alias': 'R1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2018, 11, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV-PHP.eB-EF1a-DIO-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 1, 'notes': 'blocked 0 and 1 morph trials'},
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4},
            {'date_str': '12_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '15_02_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 3},
            {'date_str': '17_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5},
        ),
        'test_sessions': (
            {'date_str': '19_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '21_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '22_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '23_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 11},
            {'date_str': '24_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4},
        ),
    },

    '4139265.4':{
        'alias': 'R2',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2018, 11, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV-PHP.eB-EF1a-DIO-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '07_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 1, 'notes': 'blocked 0 and 1 morph trials'},
            {'date_str': '07_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4},
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '12_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4},
            {'date_str': '15_02_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 5},
        ),
        'test_sessions': (
            {'date_str': '17_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 8},
            {'date_str': '18_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '19_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '20_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '21_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6},
            {'date_str': '22_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6},
        ),
    },

    '4139265.5':{
        'alias': 'R3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2018, 11, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV-PHP.eB-EF1a-DIO-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 2, 'notes': 'blocked 0 and 1 morph trials'},
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6},
            {'date_str': '12_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '15_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '17_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
        ),
        'test_sessions': (
            {'date_str': '19_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5},
            {'date_str': '21_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 9},
            {'date_str': '22_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 10},
            {'date_str': '23_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 14},
            {'date_str': '24_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7},
        ),
    },

    '4222168.1':{
        'alias': 'R4',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 3, 3),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f-WPRE',
        'notes': 'missing first day of imaging, so first training session is actually day 3',
        'training_sessions': (
            {'date_str': '25_08_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 7},
            {'date_str': '28_08_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '02_09_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4},
        ),
        'test_sessions': (
            {'date_str': '03_09_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 9},
            {'date_str': '04_09_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4},
            {'date_str': '05_09_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4},
            {'date_str': '06_09_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6},

        ),
    },

    '4343703.1':{
        'alias': 'R5',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 10, 29),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f-WPRE',
        'notes': '',
        'training_sessions': (
            {'date_str': '14_03_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 1},
            {'date_str': '16_03_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '18_03_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 17},
            {'date_str': '20_03_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6},
            {'date_str': '21_03_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 24},
        ),
        'test_sessions': (
            {'date_str': '22_03_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
        ),
    },

    '4343706': {
        'alias': 'R6',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 9, 23),
        'genotype': 'wildtype C57BL/6J',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-syn-GCaMP7f',
        'notes': 'missing training sessions',
        'training_sessions': ({'date_str': '07_02_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},),
        'test_sessions': (
            {'date_str': '09_02_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7},
            {'date_str': '10_03_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 14},
            {'date_str': '11_02_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6},
        ),
    },
    
}

frequent_sessions = {
    '4222153.1':{
        'alias': 'F1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 1, 17),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 6},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 11},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 15},
        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '16_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6},
            {'date_str': '17_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 17},
        ),
    },

    '4222153.2':{
        'alias': 'F2',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 1, 17),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 11},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 12},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7},
        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7},
            {'date_str': '16_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 9},
            {'date_str': '17_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '18_04_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 3},
        ),
    },

    '4222153.3':{
        'alias': 'F3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 1, 17),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 16},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 15},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 11},
        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6},
        ),
    },

    '4222174.1':{ # download data from google drive then push to oak
        'alias': 'F4',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2018, 10, 29),
        'genotype': 'Ai94; CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'GCaMP6f expressed from Ai94 transgenic allele',
        'notes': '',
        'training_sessions': (
            {'date_str': '04_06_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 16},
            {'date_str': '06_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 8},
            {'date_str': '08_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 13},
            {'date_str': '11_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 9},
        ),
        'test_sessions': (
            {'date_str': '13_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 8},
            {'date_str': '14_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5},
            {'date_str': '15_06_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 8},
            {'date_str': '17_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7},
            {'date_str': '18_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 21},
        ),
    },

    '4222154.1': {
        'alias': 'F5',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 1, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 5},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 4},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},

        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
            {'date_str': '16_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '17_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 14},
            {'date_str': '18_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '19_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3},
        ),
    },
    
    '4343702.1': { # download from google drive then push to oak
        'alias': 'F6',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 10, 29),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '13_03_2020', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 4},
            {'date_str': '15_03_2020', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 7},
            {'date_str': '17_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 1},
            {'date_str': '19_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
        ),
        'test_sessions': (
            {'date_str': '20_03_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4},
            {'date_str': '21_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2},
            {'date_str': '22_03_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6}, # get suite2p data from suite2p_data folder
        ),
    },

}

frequent_w_decision_sessions = {
    '4222157.4':{
        'alias': 'FD1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {},
        ),
        'test_sessions': (
            {},
        ),
    },
    
    '4222169.1': {
        'alias': 'FD2',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {},
        ),
        'test_sessions': (
            {},
        ),
    },

    '4222169.2': {
        'alias': 'FD3',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {},
        ),
        'test_sessions': (
            {},
        ),
    },

    '4222169.4': {
        'alias': 'FD4',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {},
        ),
        'test_sessions': (
            {},
        ),
    },

}

rare_mice = list(rare_sessions.keys())
frequent_mice = list(frequent_sessions.keys())
frequent_w_decision_mice = list(frequent_w_decision_sessions.keys())