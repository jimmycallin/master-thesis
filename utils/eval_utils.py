from .conll16st import scorer


def score_connectives(answers, predicted):
    scorer.evaluate(answers, predicted)
