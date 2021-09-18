import os
from pathlib import Path

from config import PipelineConfig
from pymongo import MongoClient
import numpy as np
import cv2
from copy import deepcopy


class DBManager(object):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset = {}
        self.dataset = {}

    def connect(self):
        pass

    def get_data(self, labels=None):
        pass

    def upload_data(self, data):
        pass


class MongoDBManager(DBManager):
    def __init__(self, config: PipelineConfig):
        super(MongoDBManager, self).__init__(config)
        self.db_name = self.config.mongo_db_name
        self.host = self.config.mongo_host
        self.port = self.config.mongo_port
        self.col_name = self.config.mongo_col_name
        self.client = None
        self.db = None
        self.data_col = None

    def connect(self):
        self.client = MongoClient(self.host, self.port)
        self.db = self.client[self.db_name]
        self.data_col = self.db[self.col_name]

    def upload_data(self, data):
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')
            intensity_arr = data_rec.get('intensity')
            p_mass_arr = data_rec.get('pm')
            label = data_rec.get('label')
            is_test = data_rec.get('is_test')
            file = data_rec.get('file')
            if type(intensity_arr) == np.ndarray:
                intensity_arr = intensity_arr.tolist()
            if type(p_mass_arr) == np.ndarray:
                p_mass_arr = p_mass_arr.tolist()

            self.data_col.insert_one({'file': file,
                                      'label': label,
                                      'is_test': is_test,
                                      'intensity_data': intensity_arr,
                                      'p_mass_data': p_mass_arr})

    def get_data(self, labels=None):
        if labels:
            cursor = self.data_col.find({'label': {'$in': labels}})
        else:
            cursor = self.data_col.find()

        index = 0
        for rec in cursor:
            file_ = rec.get('file')
            label_ = rec.get('label')
            is_test = rec.get('is_test')

            if 'AUG' in file_ and is_test != 2:
                continue
            data_rec = {'label': label_,
                        'file': file_,
                        'is_test': is_test,
                        'metadata': None}
            self.dataset.update({index: {'data': data_rec}})
            index += 1


class MixedData(object):
    def __init__(self, config):
        config_to_mix = deepcopy(config)
        self.db_manager_2 = MongoDBManager(config)
        config_to_mix.mongo_db_name = 'cancer_data'
        self.folder_m = config.image_folder
        self.folder_p = Path(str(self.folder_m).replace('meta', 'pept'))
        self.db_manager_1 = MongoDBManager(config_to_mix)
        self.dataset = {}

    def connect(self):
        self.db_manager_1.connect()
        self.db_manager_2.connect()

    def get_data(self):
        self.db_manager_1.get_data()
        self.db_manager_2.get_data()
        index = 0
        for key in self.db_manager_1.dataset:
            data = self.db_manager_1.dataset.get(key)
            data_rec = data.get('data')
            label = data_rec.get('label')
            is_test = data_rec.get('is_test')
            file = str(self.folder_p / str(label) / data_rec.get('file'))
            if label in (2, 5):
                continue
            if label == 4:
                label = 2
            data_rec = {'label': label,
                        'file': file,
                        'is_test': is_test,
                        'metadata': None}
            self.dataset.update({index: {'data': data_rec}})
            index += 1

        for key in self.db_manager_2.dataset:
            data = self.db_manager_2.dataset.get(key)
            data_rec = data.get('data')
            label = data_rec.get('label')
            if label == 4:
                continue
            file = str(self.folder_m / str(label) / data_rec.get('file'))
            is_test = data_rec.get('is_test')

            data_rec = {'label': label,
                        'file': file,
                        'is_test': is_test,
                        'metadata': None}
            self.dataset.update({index: {'data': data_rec}})
            index += 1


