from subprocess import call
from word_embedding_paths import word_embeddings
from os.path import exists
from os import mkdir

for c in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    name = "svmrbf-c" + str(c)
    if exists("/usit/abel/u1/jimmycallin/models/" + name):
        print("Model " + name + " already trained, skipping...")
        continue
    else:
        mkdir("/usit/abel/u1/jimmycallin/models/" + name)
    print("Training " + name)
    call(["sbatch", "--job-name", name, "--output=/usit/abel/u1/jimmycallin/models/" + name + "/stdout.txt","/usit/abel/u1/jimmycallin/architectures/abel_tune_svm.sh", name, "/word_embeddings/precompiled/glove/size=50.embeddings", str(c)])
