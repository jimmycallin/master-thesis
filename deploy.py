from os.path import exists, join
from os import makedirs
from config import config
import os, tarfile, requests

print("Changing working directory to {}".format(config['base_dir']))
os.chdir(config['base_dir'])

if not exists('resources'):
    print("Creating resources/")
    makedirs('resources')

# Download missing files
print("Downloading all files...")
for path, urls in config['download'].items():
        makedirs(path, exist_ok=True)
        for url in urls:
            file_name = url.split("/")[-1]
            full_path = join(path, file_name)
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
                    tf.extractall(path)

# Check file_exists
print("Checking if all necessary files exists...")
for f in config['check_exists']:
    if not exists(f):
        raise FileNotFoundError("Missing file: {}".format(f))
