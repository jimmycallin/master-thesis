from subprocess import call
from word_embedding_paths import word_embeddings
from os.path import exists
from os import mkdir
test_types = ['conll16st-en-03-29-16-dev']
embedding = "/word_embeddings/precompiled/glove/size=50.embeddings"
for test_type in test_types:
    for c in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        name = "svmrbf-c" + str(c)
        output_dir = "/usit/abel/u1/jimmycallin/outputs/" + name  + "-" + test_type
        if not exists(output_dir):
            mkdir(output_dir)
        else:
            print("Skipping " + name + "-" + test_type)
            continue
        print("Testing " + name + " with " + test_type)
        call(["sbatch", "--job-name", name, "--output=" + output_dir + "/stdout.txt",
              "/usit/abel/u1/jimmycallin/architectures/abel_test_svm_tune.sh", name, test_type, embedding, str(c)])
