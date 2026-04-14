import pandas as pd
import datetime

rare_sessions = {

    'R1':{
        'alias': '4139265.3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2018, 11, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV-PHP.eB-EF1a-DIO-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 1, 'notes': 'blocked 0 and 1 morph trials', 'day':1.1},
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4, 'day': 1.2},
            {'date_str': '12_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 3},
            {'date_str': '15_02_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 3, 'day': 5},
            {'date_str': '17_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '19_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 8},
            {'date_str': '21_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 9},
            {'date_str': '22_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 10},
            {'date_str': '23_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 11, 'day': 11},
            {'date_str': '24_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4, 'day': 12},
        ),
    },

    'R2':{
        'alias': '4139265.4',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2018, 11, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV-PHP.eB-EF1a-DIO-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '07_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 1, 'notes': 'blocked 0 and 1 morph trials', 'day': 1.1},
            {'date_str': '07_02_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 4, 'day': 1.2},
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 3},
            {'date_str': '12_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4, 'day': 5},
            {'date_str': '15_02_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 5, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '17_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 8, 'day': 8},
            {'date_str': '18_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 9},
            {'date_str': '19_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 10},
            {'date_str': '20_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 11},
            {'date_str': '21_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6, 'day': 12},
            {'date_str': '22_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6, 'day': 13},
        ),
    },

    'R3':{
        'alias': '4139265.5',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2018, 11, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV-PHP.eB-EF1a-DIO-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 2, 'notes': 'blocked 0 and 1 morph trials', 'day':1.1},
            {'date_str': '10_02_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6, 'day': 1.2},
            {'date_str': '12_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 3},
            {'date_str': '15_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 5},
            {'date_str': '17_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '19_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5, 'day': 8},
            {'date_str': '21_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 9, 'day': 9},
            {'date_str': '22_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 10, 'day': 10},
            {'date_str': '23_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 14, 'day': 11},
            {'date_str': '24_02_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7, 'day': 12},
        ),
    },

    'R4':{
        'alias': '4222168.1',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 3, 3),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f-WPRE',
        'notes': 'missing first day of imaging, so first training session is actually day 3',
        'training_sessions': (
            {'date_str': '25_08_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 7, 'day': 3},
            {'date_str': '28_08_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 6},
        ),
        'test_sessions': (
            {'date_str': '02_09_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4, 'day': 8},
            {'date_str': '03_09_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 9, 'day': 9},
            {'date_str': '04_09_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4, 'day': 10},
            {'date_str': '05_09_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 4, 'day': 11},
            {'date_str': '06_09_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6, 'day': 12},

        ),
    },

    'R5':{
        'alias': '4343703.1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 10, 29),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f-WPRE',
        'notes': '',
        'training_sessions': (
            # {'date_str': '14_03_2020', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 1, 'day': 1},
            {'date_str': '16_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 3},
            # {'date_str': '18_03_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 17, 'day': 5},
            # {'date_str': '20_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '21_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 24, 'day': 8}, 
            # {'date_str': '22_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 9},
        ),
    },

    'R6': {
        'alias': '4343706',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 9, 23),
        'genotype': 'wildtype C57BL/6J',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-syn-GCaMP7f',
        'notes': 'missing training sessions',
        'training_sessions': (), 
        'test_sessions': (
            # {'date_str': '07_02_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day':8},
            # {'date_str': '08_02_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day':9},
            {'date_str': '09_02_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7, 'day': 8},
            {'date_str': '10_02_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 14, 'day': 9},
            {'date_str': '11_02_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6, 'day': 10},
        ),
    },
    
}

frequent_sessions = {
    'F1':{
        'alias': '4222153.1',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 1, 17),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 6, 'day': 1},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5, 'day': 3},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 11, 'day': 5},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 15, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 8},
            {'date_str': '16_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6, 'day': 9},
            {'date_str': '17_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 17, 'day': 10},
        ),
    },

    'F2':{
        'alias': '4222153.2',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 1, 17),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 11, 'day': 1},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 12, 'day': 3},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 5},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7, 'day': 8},
            {'date_str': '16_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 9, 'day': 9},
            {'date_str': '17_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 10},
            {'date_str': '18_04_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 3, 'day': 11},
        ),
    },

    'F3':{
        'alias': '4222153.3',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 1, 17),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 16, 'day': 1},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 15, 'day': 3},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7, 'day': 5},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 11, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 6, 'day': 8},
        ),
    },

    'F4':{
        'alias': '4222174.1',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2018, 10, 29),
        'genotype': 'Ai94; CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'GCaMP6f expressed from Ai94 transgenic allele',
        'notes': '',
        'training_sessions': (
            {'date_str': '04_06_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 16, 'day': 1},
            {'date_str': '06_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 8, 'day': 3},
            {'date_str': '08_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 13, 'day': 5},
            {'date_str': '11_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 9, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '13_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 8, 'day': 8},
            {'date_str': '14_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 5, 'day': 9},
            {'date_str': '15_06_2019', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 8, 'day': 10},
            {'date_str': '17_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 7, 'day': 11},
            {'date_str': '18_06_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 21, 'day': 12},
        ),
    },

    'F5': {
        'alias': '4222154.1',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 1, 7),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '08_04_2019', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 5, 'day': 1},
            {'date_str': '10_04_2019', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 4, 'day': 3},
            {'date_str': '12_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 5},
            {'date_str': '14_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 7},

        ),
        'test_sessions': (
            {'date_str': '15_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 8},
            {'date_str': '16_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 9},
            {'date_str': '17_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 14, 'day': 10},
            {'date_str': '18_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 11},
            {'date_str': '19_04_2019', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 3, 'day': 12},
        ),
    },
    
    'F6': { 
        'alias': '4343702.1',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 10, 29),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '13_03_2020', 'scene': 'TwoTower_foraging', 'session': 1, 'scan': 4, 'day': 1},
            {'date_str': '15_03_2020', 'scene': 'TwoTower_foraging', 'session': 4, 'scan': 7, 'day': 3},
            {'date_str': '17_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 1, 'day': 5},
            {'date_str': '19_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '20_03_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 4, 'day': 8},
            {'date_str': '21_03_2020', 'scene': 'TwoTower_foraging', 'session': 2, 'scan': 2, 'day': 9}, # get suite2p data from suite2p_data folder
            {'date_str': '22_03_2020', 'scene': 'TwoTower_foraging', 'session': 3, 'scan': 6, 'day': 10}, # get suite2p data from suite2p_data folder
        ),
    },

}

frequent_w_decision_sessions = {
    'FD1':{
        'alias': '4222157.4',
        'sex': 'M',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': (
            {'date_str': '04_06_2019', 'scene': 'FreqMorph_Decision', 'session': 0, 'scan': 6, 'day': 1},
            {'date_str': '06_06_2019', 'scene': 'FreqMorph_Decision', 'session': 1, 'scan': 8, 'day': 3},
            {'date_str': '08_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 2, 'scan': 22, 'day': 5},
            {'date_str': '11_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 1, 'scan': 17, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '13_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 1, 'scan': 7, 'day': 9},
            {'date_str': '14_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 2, 'scan': 13, 'day': 10},
            {'date_str': '15_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 1, 'scan': 3, 'day': 11},
            {'date_str': '17_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 1, 'scan': 6, 'day': 12},
            {'date_str': '18_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 1, 'scan': 26, 'day': 13},
            {'date_str': '19_06_2019', 'scene': 'FreqMorph_Timeout', 'session': 1, 'scan': 5, 'day': 14},
            {'date_str': '23_06_2019', 'scene': 'TwoTower_Timeout', 'session': 1, 'scan': 17, 'day': 17},
            {'date_str': '25_06_2019', 'scene': 'TwoTower_Timeout', 'session': 1, 'scan': 9, 'day': 19},
        ),
    },
    
    'FD2': {
        'alias': '4222169.1',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': ( 
            {'date_str': '06_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 3, 'scan': 6, 'day': 1},
            {'date_str': '08_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 2, 'day': 3},
            {'date_str': '10_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 6, 'day': 5},
            {'date_str': '12_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 3, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '13_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 3, 'day': 8},
            {'date_str': '14_08_2019', 'scene': 'TwoTower_Timeout', 'session': 2, 'scan': 5, 'day': 9},
            {'date_str': '25_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 3, 'day': 15},
            {'date_str': '26_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 3, 'day': 16},
            {'date_str': '03_09_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 20, 'day': 22},
            {'date_str': '04_09_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 9, 'day': 23},
        ),
    },

    'FD3': {
        'alias': '4222169.2',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': ( 
            {'date_str': '06_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 9, 'day': 1},
            {'date_str': '08_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 5, 'day': 3},
            {'date_str': '10_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 10, 'day': 5},
            {'date_str': '12_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 6, 'day': 7},
        ),
        'test_sessions': (
        ),
    },

    'FD4': {
        'alias': '4222169.4',
        'sex': 'F',
        'date_of_birth': datetime.datetime(2019, 2, 8),
        'genotype': 'CaMKII-Cre',
        'imaging_lambda': 920,
        'functional_indicator': 'AAV1-CAG-FLEX-GCaMP6f',
        'notes': '',
        'training_sessions': ( 
            {'date_str': '06_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 11, 'day': 1},
            {'date_str': '08_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 2, 'day': 3},
            {'date_str': '10_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 13, 'day': 5},
            {'date_str': '12_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 10, 'day': 7},
        ),
        'test_sessions': (
            {'date_str': '13_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 2, 'scan': 2, 'day': 8},
            {'date_str': '14_08_2019', 'scene': 'TwoTower_Timeout', 'session': 2, 'scan': 13, 'day': 9},
            {'date_str': '25_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 7, 'day': 15},
            {'date_str': '26_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 7, 'day': 16},
            {'date_str': '27_08_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 6, 'day': 17},
            {'date_str': '03_09_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 24, 'day': 22},
            {'date_str': '04_09_2019', 'scene': 'TwoTower_noTimeout', 'session': 1, 'scan': 12, 'day': 23},
        ),
    },

}

rare_mice = list(rare_sessions.keys())
frequent_mice = list(frequent_sessions.keys())
frequent_w_decision_mice = list(frequent_w_decision_sessions.keys())

# add parsed datetime (12:00) to each session dict (uses 'date_str' or fallback 'date')
def _add_session_datetimes(container):
    def process_item(item):
        if item is None:
            return
        if isinstance(item, dict):
            date_str = item.get('date_str') or item.get('date')
            if date_str:
                try:
                    parsed = datetime.datetime.strptime(date_str, '%d_%m_%Y')
                    item['datetime'] = datetime.datetime(parsed.year, parsed.month, parsed.day)
                except Exception:
                    pass
        elif isinstance(item, (list, tuple)):
            for it in item:
                process_item(it)

    for mouse in container.values():
        sessions = mouse['training_sessions'] + mouse['test_sessions']
        for entry in sessions:
            process_item(entry)


_add_session_datetimes(rare_sessions)
_add_session_datetimes(frequent_sessions)