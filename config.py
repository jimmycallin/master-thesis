config = {
    "author": "Jimmy Callin",
    "base_dir": "/Users/jimmy/dev/edu/master-thesis/",
    "train_dir": "resources/conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-train/",
    "test_dir": "resources/conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-dev/",
    "download": {"resources/brown_clusters/": ["http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/README.txt",
                                               "http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt",
                                               "http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/brown-rcv1.clean.tokenized-CoNLL03.txt-c320-freq1.txt",
                                               "http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt",
                                               "http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt"],
                 "resources/mpqa_subjectivity_lexicon": ["http://www.cs.brandeis.edu/~clp/conll16st/data/mpqa_subjectivity_lexicon.tar.gz"],
                 "resources/": ["http://verbs.colorado.edu/verb-index/vn/verbnet-3.2.tar.gz"]},
    "check_exists": ["resources/conll16st-en-zh-dev-train_LDC2016E50",
                     "resources/GoogleNews-vectors-negative300.bin",
                     "resources/new_vn"]
}
