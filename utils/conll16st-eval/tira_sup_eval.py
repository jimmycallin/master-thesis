#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The evaluator used on the TIRA evaluation plaform for CoNLL 2016 Shared Task
(Supplementary task of discourse relation sense classification only)

"""
import json
import sys
from scorer import evaluate, evaluate_sense
from partial_scorer import partial_evaluate
from validator import validate_relation_list, identify_language
from tira_eval import write_proto_text, write_results

def use_gold_standard_types(sorted_gold_relations, sorted_predicted_relations):
    for gr, pr in zip(sorted_gold_relations, sorted_predicted_relations):
        if gr['ID'] != pr['ID']:
            print >> sys.stderr, 'ID mismatch. Make sure you copy the ID from gold standard'
            exit(1)
        pr['Type'] = gr['Type']


def main(args):
    input_dataset = args[1]
    input_run = args[2]
    output_dir = args[3]

    gold_relations = [json.loads(x) for x in open('%s/relations.json' % input_dataset)]
    filter_ids = set()
    for rel in gold_relations:
        if rel['Type'] in ("Explicit", "AltLex"):
            filter_ids.add(rel['ID'])
    gold_relations = [x for x in gold_relations if x['ID'] not in filter_ids]
    predicted_relations = [json.loads(x) for x in open('%s/output.json' % input_run)]
    predicted_relations = [x for x in predicted_relations if x['ID'] not in filter_ids]
    if len(gold_relations) != len(predicted_relations):
        err_message = 'Gold standard has % instances; predicted %s instances' % \
                (len(gold_relations), len(predicted_relations))
        print >> sys.stderr, err_message
        exit(1)

    language = identify_language(gold_relations)
    all_correct = validate_relation_list(predicted_relations, language)
    if not all_correct:
        print >> sys.stderr, 'Invalid format'
        exit(1)

    gold_relations = sorted(gold_relations, key=lambda x: x['ID'])
    predicted_relations = sorted(predicted_relations, key=lambda x: x['ID'])
    use_gold_standard_types(gold_relations, predicted_relations)

    output_file = open(output_dir, 'w')
    print 'Evaluation for all discourse relations'
    write_results('All', evaluate(gold_relations, predicted_relations), output_file)

    print 'Evaluation for explicit discourse relations only'
    explicit_gold_relations = [x for x in gold_relations if x['Type'] == 'Explicit']
    explicit_predicted_relations = [x for x in predicted_relations if x['Type'] == 'Explicit']
    write_results('Explicit only', \
        evaluate(explicit_gold_relations, explicit_predicted_relations), output_file)

    print 'Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)'
    non_explicit_gold_relations = [x for x in gold_relations if x['Type'] != 'Explicit']
    non_explicit_predicted_relations = [x for x in predicted_relations if x['Type'] != 'Explicit']
    write_results('Non-explicit only', \
        evaluate(non_explicit_gold_relations, non_explicit_predicted_relations), output_file)

    sense_cm = evaluate_sense(non_explicit_gold_relations, non_explicit_predicted_relations)
    for i in xrange(sense_cm.alphabet.size()):
        precision, recall, f1 = sense_cm.get_prf_for_i(i)
        label = sense_cm.alphabet.get_label(i)
        if label != sense_cm.NEGATIVE_CLASS:
            output_file.write('measure {\n key: "Non-explicit ' + label + ' Precision"\n value: "' + str(precision) + '"\n}\n')
            output_file.write('measure {\n key: "Non-explicit ' + label + ' Recall"\n value: "' + str(recall) + '"\n}\n')
            output_file.write('measure {\n key: "Non-explicit ' + label + ' F1"\n value: "' + str(f1) + '"\n}\n')

    output_file.close()

if __name__ == '__main__':
    main(sys.argv)

