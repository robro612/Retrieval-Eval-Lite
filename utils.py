from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import bm25s
import faiss
import ir_datasets
import pandas as pd
import polars as pl
import pyterrier as pt
import Stemmer
import torch
from pyterrier.measures import AP, R, RR, nDCG, Success
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import os


class DataTypes:
    # documents are [docno, text]
    Document = namedtuple("Document", ["docno", "text"])
    # queries are [qid, query]
    Query = namedtuple("Query", ["qid", "query"])
    # qrels are [qid, docno, label]
    Qrel = namedtuple("Qrel", ["qid", "docno", "label"])
    # retrieval results are a pd.DataFrame with [qid, docno, score, rank]
    Result = namedtuple("Result", ["qid", "docno", "score", "rank"])

PT_METRICS = [
    *[nDCG@k for k in (10, 100, 1000)],
    *[R@k for k in (1, 10, 100, 1000)]
    *[RR@k for k in (10, 100, 1000)],
    *[Success@k for k in (1, 5, 10, 50, 100)],
]

RETRIEVER_OUTPUT_FILENAME = "retriever_output.tsv"
RESULTS_FILENAME = "results.tsv"

class Retriever(pt.Transformer, ABC):

    @abstractmethod
    def index(self, corpus: pd.DataFrame, index_dir: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def load_index(self, index_dir: str) -> None:
        pass


class BM25s(Retriever):
    """
    A Pyterrier transformer that implements BM25 using the BM25S library
    When instantiated, will index the corpus (and optionally save it) or load a previously build index
    """

    def __init__(
        self,
        corpus: pd.DataFrame,
        lang="en",
        stemmer: Optional[Stemmer.Stemmer] = None,
        index_dir: Optional[str] = None,
        load_index: bool = False,
    ):
        self.corpus = corpus
        self.lang = lang
        self.stemmer = stemmer
        self.index_dir = index_dir

        self.retriever = None

    def index(self, corpus: pd.DataFrame, index_dir: Optional[str] = None) -> None:

        self.retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(
            corpus["text"].to_list(), stopwords=self.lang, stemmer=self.stemmer
        )
        self.retriever.index(corpus_tokens)

        if self.index_dir is not None:
            print(f"Saving index to {self.index_dir}")
            self.retriever.save(self.index_dir)

    def load_index(self, index_dir: str) -> None:
        self.retriever = bm25s.BM25.load(index_dir)

    def transform(self, queries: pd.DataFrame, k: int = 1000) -> pd.DataFrame:
        if self.retriever is None:
            raise ValueError(
                "Retriever is not initialized - call self.index() or self.load_index() to load an index before trying to retrieve."
            )

        query_token_ids = bm25s.tokenize(
            queries["query"].to_list(), stopwords=self.lang, stemmer=self.stemmer
        )
        docnos, scores = self.retriever.retrieve(
            query_token_ids,
            corpus=self.corpus["docno"].to_list(),
            k=k,
        )
        results = pd.DataFrame(
            [
                DataTypes.Result(qid=qid, docno=docno, score=score, rank=rank)
                for qid, docnos_i, scores_i in zip(queries["qid"], docnos, scores)
                for rank, (docno, score) in enumerate(zip(docnos_i, scores_i), start=1)
            ]
        )
        return results


class DenseRetriever(Retriever):
    """
    A pyterrier transformer that indexes (Faiss) and searches with a Sentence Transformer
    When instantiated, will index the corpus (and optionally save it) or load a previously build index
    """

    INDEX_FILENAME = "faiss_flat_index.bin"
    DOCNOS_FILENAME = "docids.tsv"

    def __init__(
        self,
        metric=faiss.METRIC_INNER_PRODUCT,
        max_seq_length: Optional[int] = None,
        **sentence_transformer_kwargs: Dict[str, Any],
    ):
        self.metric = metric
        self.model = SentenceTransformer(**sentence_transformer_kwargs)

        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def index(
        self,
        corpus: pd.DataFrame,
        index_dir: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:

        self.docnos = corpus.docno

        # Create a FAISS IndexIDMap to store both vectors and their IDs
        self.faiss_index = faiss.index_factory(
            self.model.get_sentence_embedding_dimension(), "IDMap,Flat", self.metric
        )

        self.model.eval()

        # Process the corpus in batches
        for i in tqdm(range(0, len(corpus), batch_size), desc="Indexing"):
            batch = corpus.iloc[i : i + batch_size]

            # Extract text and IDs from the batch
            texts = batch["text"].tolist()
            doc_ids = batch.index.tolist()
            # Encode the texts into embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=(self.metric == faiss.METRIC_INNER_PRODUCT),
                batch_size=batch_size,
            ).astype(np.float32)

            # Add the embeddings to the index along with their IDs
            self.faiss_index.add_with_ids(embeddings, np.array(doc_ids))

        # Save the index if a path is provided
        if index_dir:
            self._save_index(index_dir)

        print(f"Indexed {len(corpus)} documents.")

    def _save_index(self, index_dir: str) -> None:
        faiss.write_index(
            self.faiss_index, os.path.join(index_dir, DenseRetriever.INDEX_FILENAME)
        )

        self.docnos.to_csv(
            os.path.join(index_dir, DenseRetriever.DOCNOS_FILENAME),
            sep="\t",
            header=True,
            index=False,
        )

    def load_index(self, index_dir: str) -> None:
        self.faiss_index = faiss.read_index(
            os.path.join(index_dir, DenseRetriever.INDEX_FILENAME)
        )
        self.docnos = pd.read_csv(
            os.path.join(index_dir, DenseRetriever.DOCNOS_FILENAME), sep="\t"
        ).docno

    def transform(self, queries: pd.DataFrame, k=1000) -> pd.DataFrame:

        if self.faiss_index is None:
            raise ValueError(
                "Index is not initialized - call self.index() or self.load_index() to load an index before trying to retrieve."
            )

        q_embs = self.model.encode(
            queries["query"], show_progress_bar=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, idxs = self.faiss_index.search(q_embs, k=k)

        results = pd.DataFrame(
            DataTypes.Result(qid=qid, docno=self.docnos[idx], score=score, rank=rank)
            for qid, q_scores, q_idxs in zip(queries["qid"], scores, idxs)
            for rank, (score, idx) in enumerate(zip(q_scores, q_idxs), start=1)
        )
        return results


def load_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads a dataset from ir_datasets into corpus, queries, and qrels with data formats
    according to DataTypes
    """
    dataset = ir_datasets.load(dataset_name)

    queries = pd.DataFrame(dataset.queries_iter())[["query_id", "text"]].rename(
        columns={"query_id": "qid", "text": "query"}
    )

    qrels = (
        pd.DataFrame(dataset.qrels_iter())
        .iloc[:, :3]
        .rename(columns={"query_id": "qid", "doc_id": "docno", "relevance": "label"})
    )

    corpus = pd.DataFrame(dataset.docs_iter())

    if "title" in corpus.columns:
        corpus["text"] = corpus["title"] + " | " + corpus["text"]

    corpus = corpus[["doc_id", "text"]].rename(columns={"doc_id": "docno"})

    return corpus, queries, qrels


def generate_results(
    corpus: pd.Dataframe,
    queries: pd.Dataframe,
    qrels: pd.Dataframe,
    save_results_dir: str,
    results_name: str,
    retriever: Optional[type | Retriever],
    load_index: bool = False,
    index_dir: Optional[str] = None,
    **retriever_kwargs: Dict[str, Any],
) -> Tuple[pd.Dataframe, pd.Dataframe]:

    if isinstance(retriever, type):
        retriever = retriever(**retriever_kwargs)

    if load_index:
        retriever.load_index(index_dir=index_dir)
    else:
        retriever.index(corpus, index_dir=index_dir)

    retrieval_output = dr.transform(queries, k=1000)

    eval_metrics = pt.Experiment(
        [retrieval_output],
        queries,
        qrels,
        eval_metrics=PT_METRICS,
        names=[results_name],
    )

    retrieval_output.to_csv(os.path.join(save_results_dir, RETRIEVER_OUTPUT_FILENAME), sep="\t", index=False, header=True)
    eval_metrics.to_csv(os.path.join(save_results_dir, RESULTS_FILENAME), sep="\t", index=False, header=True)

    return eval_metrics, retrieval_output

if __name__ == "__main__":

    dr = DenseRetriever(
        model_name_or_path="jinaai/jina-embeddings-v2-base-en-flash",
        device="cuda",
        max_seq_length=512,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    )

    corpus, queries, qrels = load_dataset("beir/nq")

    print("Indexing")
    dr.index(
        corpus,
        index_dir="index/nq/full_corpus/jina-embeddings-v2-base-en-flash",
        batch_size=8192,
    )

    print("Loading Index")
    dr.load_index(index_dir="index/nq/full_corpus/jina-embeddings-v2-base-en-flash")

    results = dr.transform(queries, k=1000)

    pt.init()

    experiment = pt.Experiment(
        [dr],
        queries,
        qrels,
        eval_metrics=[nDCG @ 10, R @ 1000],
        names=["Dense Retriever"],
    )
