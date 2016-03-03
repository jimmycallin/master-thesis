from os.path import exists, join
from os import makedirs
import os, tarfile, requests, yaml, hashlib

def read_config(path):
    with open(path) as f:
        return yaml.load(f)

def compute_sha(path):
    with open(path, 'rb', buffering=16*1024*1024) as f:
        sha = hashlib.sha1()
        for buffer in f:
            sha.update(buffer)
    return sha.hexdigest()

config = read_config('config.yaml')

print("Changing working directory to {}".format(config['base_dir']))
os.chdir(config['base_dir'])

if not exists('resources'):
    print("Creating resources/")
    makedirs('resources')

# Download missing files
print("Downloading all files...")
for download in config['download']:
    makedirs(download['to'], exist_ok=True)
    for url in download['from']:
        file_name = url.split("/")[-1]
        full_path = join(download['to'], file_name)
        if exists(full_path):
            print("{} already exists, skipping".format(full_path))
            continue
        print("Downloading {} to {}".format(file_name, full_path))
        with open(full_path, "wb") as f:
            response = requests.get(url, stream=True)
            for block in response.iter_content(1024):
                f.write(block)
        print("{} downloaded".format(file_name))
        if str.endswith(file_name, ".tar.gz"):
            print("Extracting...")
            with tarfile.open(full_path) as tf:
                tf.extractall()

# Check file_exists
print("Checking if all necessary files exists...")
for f in config['check_exists']:
    if 'shasum' in f:
        print('Calculating shasum for {}...'.format(f['name']))
        shasum = compute_sha(f['path'])
    if not exists(f['path']):
        raise FileNotFoundError("Missing file: {}".format(f))
    elif 'shasum' in f and shasum != f['shasum']:
        raise ValueError('Shasum does not match file {}'.format(f['path']))

print('Deployed successfully!')
