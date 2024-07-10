
## Directory/Artifact Structure

### Indices (file name(s) depend on model type - BM25/DR)

```
index/dataset/subsample_method/model/
```
### Results (evaluation output and raw retrieval ranking/scores)

```
results/dataset/subsample_method/model/results.tsv
results/dataset/subsample_method/model/retriever_output.tsv

```

## Experiment Guidelines
- Always use the following metrics:
    - Recall@{1,10,100,1000}
    - Success@{1,5,10,100}
    - RR@{10, 100, 1000}
    - nDCG@{10, 100, 1000}
- Save all artifacts as .tsv without index and with header