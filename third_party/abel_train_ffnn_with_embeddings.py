from subprocess import call
from word_embedding_paths import word_embeddings
from os.path import exists

for embedding in word_embeddings:
	name = "ffnn-" + embedding["name"]
	if exists("/usit/abel/u1/jimmycallin/models/" + name):
		print("Model " + name + " already trained, skipping...")
		continue
	print("Training " + name)
	call(["sbatch", "--job-name", name, "--output=/usit/abel/u1/jimmycallin/models/" + name + "/stdout.txt","/usit/abel/u1/jimmycallin/third_party/abel_train_ffnn.sh", name, embedding["path"]])
