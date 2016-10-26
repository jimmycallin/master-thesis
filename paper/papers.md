## Pacheco et al., “Adapting Event Embedding for Implicit Discourse Relation Recognition.”

- The paper uses a simple feed-forward NN and something called Event Embeddings and reaches competitive results for implicit sense classification.
- It uses two different classifiers for sense identification: a SVM with linear kernel for explicit relations with SOTA features, and a multi-layer neural network for implicit relations.
- Instead of using word sequences as input to train the embeddings, they use event chains extracted by connecting events with co-referencing entities.
- Word event vectors are found useful.
-
- Quotes:
    - "Last year, most submitted systems used algorithms traditionally applied for this task, such as SVMs and MaxEnt classifiers learned over binary features as input representation. This included the best performing system which reached an accuracy of 34.45 in the test data and an accuracy of 36.29 in the blind test data for implicit relations (Xue et al. 2015, Qiang and Lan 2015)"

## Mihaylov, Todor, and Anette Frank. “Discourse Relation Sense Classification Using Cross-Argument Semantic Similarity Based on Word Embeddings.” ACL 2016, 2016, 100.

- The paper uses a Logistic Regression classifier and gets highest explicit sense classiication in blind test, and word2vec with CNN to achieve competitive to get fourth highest implicit sense classification F1. The LR classifier uses similarity features from WE.
- It has a good list of previous work in last year's shared task. Use this when writing your own summary.
- The CNN experimented with dependency-based word embeddings from Levy and Goldberg 2014. They got slightly worse results on the dev set
- CNN CANDIDATE


## Kido, Yusuke, and Akiko Aizawa. “Discourse Relation Sense Classification with Two-Step Classifiers.” ACL 2016, 2016, 129.

- The paper uses a two-step classifier to gain competitive results (8th) with an SVM and MaxEnt classifier.
- It has training data analysis numbers for connective tokens per class (figure 1)
- It replaces unknown connective words with known connective words by looking up which connective word is the closest according to word2vec.
- Notes the difficulty of disambiguating Comparison.Concession from Contrast.

## Weiss, Gregor, and Marko Bajec. “Discourse Sense Classification from Scratch Using Focused RNNs.” ACL 2016 1, no. 100 (2016): 50.
- RNN CANDIDATE

## Schenk, Niko, Christian Chiarcos, Kathrin Donandt, Samuel Rönnqvist, Evgeny A. Stepanov, and Giuseppe Riccardi. “Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling.” ACL 2016, 2016, 41.

- A modular FFNN using WE and dependencies. Achieves pretty good results for being generic.
- FFNN CANDIDATE
