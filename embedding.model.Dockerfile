FROM semitechnologies/transformers-inference:custom
RUN MODEL_NAME=aari1995/German_Semantic_STS_V2 ./download.py