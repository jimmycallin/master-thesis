from subprocess import call
from word_embedding_paths import word_embeddings
from os.path import exists
from os import mkdir

for embedding in word_embeddings:
    name = "logreg-" + embedding["name"]
    if exists("/usit/abel/u1/jimmycallin/models/" + name):
        print("Model " + name + " already trained, skipping...")
        continue
    else:
        mkdir("/usit/abel/u1/jimmycallin/models/" + name)
    print("Training " + name)
    call(["sbatch", "--job-name", name, "--output=/usit/abel/u1/jimmycallin/models/" + name + "/stdout.txt","/usit/abel/u1/jimmycallin/third_party/abel_train_logreg.sh", name, embedding["path"]])
