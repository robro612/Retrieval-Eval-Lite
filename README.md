
## Directory/Artifact Structure

### Indices (file name(s) depend on model type - BM25/DR)

```
index/${dataset}/${subsample_method}/${retriever}
```
### Results (evaluation output and raw retrieval ranking/scores)

```
results/${dataset}/${subsample_method}/${retriever}/results.tsv
results/${dataset}/${subsample_method}/${retriever}/retriever_output.tsv

```

Where `subsample_method` is named:
- `full` for full corpus
- `${"_".join(models)}_top_${top_k}` for subsampled by union of topk from models 

## Experiment Guidelines
- Always use the following metrics:
    - Recall@{1,10,100,1000}
    - Success@{1,5,10,100}
    - RR@{10, 100, 1000}
    - nDCG@{10, 100, 1000}
- Save all artifacts as .tsv without index and with header