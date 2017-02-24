from subprocess import call
from word_embedding_paths import word_embeddings
from os.path import exists
from os import mkdir

test_types = ['conll15st-en-03-29-16-blind-test', 'conll16st-en-03-29-16-dev', 'conll16st-en-03-29-16-test', 'conll16st-en-03-29-16-trial']

#word_embeddings = [{'name': 'reproduce-1', 'path': 'none-1'}, {'name': 'reproduce-2', 'path': 'none-2'}, {'name': 'reproduce-3', 'path': 'none-3'}, {'name': 'reproduce-4', 'path': 'none-4'}]

for test_type in test_types:
	for embedding in word_embeddings:
		name = "staticrnn-" + embedding["name"]
		output_dir = "/usit/abel/u1/jimmycallin/outputs/" + name  + "-" + test_type
		if not exists(output_dir):
			mkdir(output_dir)
		else:
			print("Skipping " + name + "-" + test_type + "...")
			continue
		print("Testing " + name + " with " + test_type)
		call(["sbatch", "--job-name", name, "--output=" + output_dir + "/stdout.txt",
			  "/usit/abel/u1/jimmycallin/architectures/abel_test_rnn.sh", name, test_type, embedding["path"]])
