# Background

Most work in computational linguistics currently makes simplified assumptions regarding the atomicity of sentences. They are rarely linked, provide little information on a document level, and at best we have some explicit domain knowledge regarding their current context. This is necessary for several reasons:

- We have not really had a well-established framework for discourse analysis.
- Previous theories have been competing, lacking in properties in several ways
- It has been unclear how to measure the quality of a good theoretical discourse framework, making it difficult to judge what discourse framework has been the best

Regardless of these properties, all of which still can be argued to be relevant, we have today at least some data to work with. A subset of Penn Treebank has been annotated with a set of discourse connectives, giving us a resource to work with to learn more about the computability of these connectives. This is a good start.

Penn Discourse Treebank follows the lexically grounded predicate-argument approach as proposed in _Webber (2004)_. It covers the subset containing Wall Street Journal articles from the Penn Treebank, making up approximately one million tokens. When a connective explicitly appears, it will be syntactically connected to the _Arg2_ argument of the discourse structure. _Arg1_ is the other one. Due to _Arg2_ being syntactically bounded to connective, it is easy to automatically classify _Arg2_. _Arg1_ is more difficult, and Lin et al (2012) solves this by

PDTB annotates each structure with types of discourse relations according to a three level hierarchy, where the first level is made up of four classes: temporal, contingency, comparison, expansion. Each class has a second level of in total 16 types to provide a more fine-grained classification. Due to the third level being considered too fine-grained, it is ignored in this work.

## Explicit vs implicit discourse connectives

- Explicit
But, however, although, so

I want to go to New York, but I already booked a flight.  (--> I am not going to New York.)
I want to go to New York, so I already booked a flight. (--> I am going to New York.)


- Implicit

- EntRel, AltRel?


## Usefulness

Discourse parsing will find its usefulness when current methods reach their potential given the current method of only looking within the immediate local context. When structure starts to become more important, as is already the case for e.g. QA systems but also in machine translation where pronouns may depend upon antecedents beyond the local context. Furthermore, experiments have been made where discourse structure measures coherence of text. See Lin, Ng, Kan (2011) Recognizing implicit discourse relations in the penn discourse treebank. Also summarization. 
