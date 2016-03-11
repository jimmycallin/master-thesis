import shelve
from os.path import join
from time import time
import git  # pylint: disable=E0401
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime


class Project():
    def __init__(self,
                 project_name,
                 project_config,
                 logger=None,
                 force_clean_repo=False,
                 mongodb_uri=None):
        if force_clean_repo:
            _assert_repo_is_clean(os.getcwd)
        self.db_uri = mongodb_uri
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client['projects'][project_name]
        self.project_name = project_name
        self.project_config = project_config
        self.logger = logger


    @property
    def project_config(self):
        config = self.db['project_config'].find_one({}, {'project_name': False, '_id': False})
        if config is None:
            return {}
        return config

    @project_config.setter
    def project_config(self, val):
        assert isinstance(val, dict), "Project config must be dict instance."
        self.db.project_config.delete_one({'project_name': self.project_name})
        if isinstance(val, dict) and val != {}:
            self.db.project_config.update_one({'project_name': self.project_name}, {'$set': val}, upsert=True)

    def set_params(self, **kwargs):
        for param, val in kwargs.items():
            self.db.insert_one({param: val})

    @property
    def experiments(self):
        experiments = [Experiment(self, i['_id']) for i in self.db['experiments'].find({}, {'_id': True})]
        return experiments

    def get_experiment(self, experiment_id):
        exp_id = self.db.experiments.find_one({'_id': experiment_id}, {'_id': True})
        if exp_id is None:
            raise ValueError("Experiment with ID {} does not exist in {}.".format(experiment_id,
                                                                                  self.project_name))
        return Experiment(self, exp_id)

    def get_latest_experiment(self):
        if len(self.experiments) == 0:
            raise ValueError("There are no experiments in this project.")

        return self.experiments[-1]

    def new_experiment(self, config):
        return Experiment(self, config=config)

    def __repr__(self):
        return "Project: {}".format(self.project_name)

class Experiment():
    def __init__(self, project, experiment_id=None, config=None, tags=None):
        self.project = project
        if isinstance(experiment_id, str):
            experiment_id = ObjectId(experiment_id)

        if experiment_id:
            self.experiment_id = experiment_id
            instance = project.db.experiments.find_one({'_id': experiment_id})
            if instance is None:
                raise ValueError("Experiment with id {} does not exist".format(experiment_id))
        else:
            self.experiment_id = self.project.db.experiments.insert_one({}).inserted_id
            self.time_added = datetime.now()
            self.tags = tags
            self.config = config

    def to_dict(self):
        return self._experiment_state

    def set_params(self, **kwargs):
        self._update_db(kwargs)

    def delete_experiment(self):
        self.project.db.experiments.delete_one({'_id': self.experiment_id})

    @property
    def _experiment_state(self):
        return self.project.db.experiments.find_one({'_id': self.experiment_id})

    def _update_db(self, val_dict):
        self.project.db.experiments.update_one({'_id': self.experiment_id}, {'$set': val_dict})

    def insert_parameter(self, val):
        assert isinstance(val, dict), "Parameter must be of dict instance"

    @property
    def config(self):
        project_config = self.project.project_config
        exp_config = self._experiment_state.get('config', {})

        return {**project_config, **exp_config}

    @config.setter
    def config(self, val):
        if val is None:
            val = {}

        exp_config = self._experiment_state.get('config', {})

        for ex, ey in val.items():
            if ex in self.project.project_config and self.project.project_config[ex] != ey:
                exp_config[ex] = ey
            elif ex not in self.project.project_config:
                exp_config[ex] = ey
        self._update_db({'config': exp_config})

    @property
    def tags(self):
        return self._experiment_state.get('tags', [])

    @tags.setter
    def tags(self, val):
        if val is None:
            val = []
        assert isinstance(val, list), "Tags must be list instance"
        self._update_db({'tags': val})

    @property
    def start_time(self):
        return self._experiment_state.get('start_time', None)

    @start_time.setter
    def start_time(self, val):
        self._update_db({'start_time': val})

    @property
    def time_added(self):
        return self._experiment_state.get('time_added', None)

    @time_added.setter
    def time_added(self, val):
        self._update_db({'time_added': val})


    @property
    def stop_time(self):
        return self._experiment_state.get('stop_time', None)

    @stop_time.setter
    def stop_time(self, val):
        self._update_db({'stop_time': val})

    @property
    def time_elapsed(self):
        start = self._experiment_state.get('start_time', None)
        stop = self._experiment_state.get('stop_time', None)
        if start is None:
            raise RuntimeError("Experiment hasn't started yet.")
        if stop is None:
            return datetime.now() - start
        return stop - start

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, type_, value, traceback):
        self.stop_time = datetime.now()

    def __repr__(self):
        return "Experiment {}/{}: {}".format(self.project.project_name,
                                             self.experiment_id,
                                             self.time_added.strftime("%Y-%m-%d %H:%M:%S"))


def list_projects(mongodb_uri=None):
    """
    This lists all projects available in 'projects' db in your Mongo db.
    """
    if mongodb_uri is None:
        client = MongoClient()
    else:
        client = MongoClient(mongodb_uri)
    return list(set([coll.split(".")[0]
                     for coll in client.projects.collection_names(include_system_collections=False)]))


def _assert_repo_is_clean(path):
    repo = git.Repo(path)
    if repo.is_dirty():
        raise RuntimeError(
            "Your repository is dirty, please commit changes to any modified files.")
    return True
