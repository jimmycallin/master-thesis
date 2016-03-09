import shelve
from os.path import join
from time import time
import git  # pylint: disable=E0401
import os

class Project():
    def __init__(self,
                 project_name,
                 config,
                 project_dir=".",
                 logger=None,
                 force_clean_repo=False):
        self.config = config
        self.project_name = project_name
        self.project_dir = project_dir
        self.db = None
        self.logger = logger
        if force_clean_repo:
            assert_repo_is_clean(os.getcwd())

    def _connect_db(self):
        os.makedirs(join(self.project_dir, 'results'), exist_ok=True)
        self.db = shelve.open(self.project_store_path, writeback=True)

    def _close_db(self):
        self.db.close()

    @property
    def experiments(self):
        self._connect_db()
        self._setup_project()
        experiments = self.db[self.project_name]['experiments']
        self._close_db()
        return experiments

    def _setup_project(self):
        if self.project_name not in self.db:  # pylint: disable=E1135
            self.db[self.project_name] = {'config': self.config, 'experiments': {}} # pylint: disable=E1136

    @property
    def project_store_path(self):
        return join(self.project_dir, 'results', self.project_name)

    def new_experiment(self, config):
        new_key = sorted(self.experiments.keys(), reverse=True)[0] + 1
        return Experiment(new_key, self, self.config)

    def store_experiment(self, experiment, **kwargs):
        commit_id = git.Repo(os.getcwd()).commit().hexsha
        exp_instance = {'config': experiment.config,
                        'time_elapsed': experiment.time_elapsed,
                        'commit_id': commit_id, **kwargs}
        self.db[self.project_name]['experiments'][experiment.experiment_id] = exp_instance
        return exp_instance

class Experiment():
    def __init__(self, experiment_id, project, config=None):
        self.experiment_id = experiment_id
        self.project = project
        self.start_time = time()
        self.stop_time = None
        self.exp_instance = None
        if config is None:
            self.config = {}

        experiment_specific = {}
        for ex, ey in config.items():
            if ex in self.project.config and self.project.config[ex] != ey:
                experiment_specific[ex] = ey
            elif ex not in self.project.config:
                experiment_specific[ex] = ey
        self.config = experiment_specific

    def log(self, **kwargs):
        self.exp_instance = self.project.store_experiment(experiment=self, **kwargs)
        return self.exp_instance

    def stop_timer(self):
        self.stop_time = time()
        return self.time_elapsed

    def __enter__(self):
        self.project._connect_db()
        self.project._setup_project()
        return self

    def __exit__(self, type_, value, traceback):
        self.stop_timer()
        self.project._close_db()

    @property
    def time_elapsed(self):
        if self.stop_time:
            return self.stop_time - self.start_time
        else:
            return time() - self.start_time

def list_experiments(db_path, project):
    with shelve.open(db_path) as db:
        if project not in db:
            return []
        else:
            return db[project]

def get_latest_experiment(db_path, project_name):
    with shelve.open(db_path) as db:
        if project_name not in db:
            raise ValueError("{} not available in {}".format(project_name, db_path))
        else:
            return db[project_name]['experiments'][-1]

def assert_repo_is_clean(path):
    repo = git.Repo(path)
    if repo.is_dirty():
        raise RuntimeError(
            "Your repository is dirty, please commit changes to any modified files.")
    return True
