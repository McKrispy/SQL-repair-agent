import math
import re

TAGS=("where","group","having","distinct","limit","desc","order_count","intersect","except")

def features(question):
    words=re.findall(r"[a-z]+|[0-9]+",question.lower())
    words=["#num" if w.isdigit() else w for w in words]
    return set(words+[" ".join(pair) for pair in zip(words,words[1:])])

def logits(model,question):
    f=features(question)
    return {tag:intercept+sum(weights.get(k,0.) for k in sorted(f)) for tag,(intercept,weights) in model.items()}

def grade(values, flags):
    # Omitting the Bernoulli normalizers is exact for a fixed question.
    return sum(max(-4,min(4,values[t])) for t in flags)

