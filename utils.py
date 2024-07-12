from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Set

import bm25s
import faiss
import ir_datasets
import pandas as pd
import polars as pl
import pyterrier as pt
import torch
import argparse
from collections import defaultdict
from pyterrier.measures import AP, R, RR, nDCG, Success
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from Stemmer import Stemmer
from loguru import logger
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
    *[nDCG @ k for k in (10, 100, 1000)],
    *[R @ k for k in (1, 10, 100, 1000)],
    *[RR @ k for k in (10, 100, 1000)],
    *[Success @ k for k in (1, 5, 10, 50, 100)],
]

RETRIEVER_OUTPUT_FILENAME = "retriever_output.tsv"
RESULTS_FILENAME = "results.tsv"
FAISS_INDEX_FILENAME = "faiss_flat_index.bin"
DOCNOS_FILENAME = "docids.tsv"

DATASET_TO_IRDS_NAME = {
    "nq": "beir/nq",
    "nfcorpus": "beir/nfcorpus/dev",
    "hotpotqa": "beir/hotpotqa/dev",
    "msmarco": "beir/msmarco/dev",
}


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
        lang="en",
        stemmer: Optional[Stemmer] = None,
    ):
        self.lang = lang
        self.stemmer = stemmer

        self.retriever = None

    def index(self, corpus: pd.DataFrame, index_dir: Optional[str] = None) -> None:

        self.docnos = corpus.docno
        self.retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(
            corpus["text"].to_list(), stopwords=self.lang, stemmer=self.stemmer
        )
        self.retriever.index(corpus_tokens)

        if index_dir is not None:
            self._save_index(index_dir)

    def _save_index(self, index_dir: str) -> None:
        logger.info(f"Saving index and docnos to {index_dir}")
        os.makedirs(index_dir, exist_ok=True)

        self.retriever.save(index_dir)

        self.docnos.to_csv(
            os.path.join(index_dir, DOCNOS_FILENAME),
            sep="\t",
            header=True,
            index=False,
        )

    def load_index(self, index_dir: str) -> None:
        self.retriever = bm25s.BM25.load(index_dir)

        self.docnos = pd.read_csv(
            os.path.join(index_dir, DOCNOS_FILENAME), sep="\t"
        ).docno

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
            corpus=self.docnos.to_list(),
            k=min(k, len(self.docnos)),
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

    def __init__(
        self,
        faiss_metric=faiss.METRIC_INNER_PRODUCT,
        max_seq_length: Optional[int] = None,
        batch_size: int = 32,
        prompts: Dict[str, str] = {"query" : "", "document" : ""},
        **sentence_transformer_kwargs: Dict[str, Any],
    ):
        self.faiss_metric = faiss_metric
        self.prompts = prompts
        self.model = SentenceTransformer(**{
            "prompts" : self.prompts,
            **sentence_transformer_kwargs,
        })
        self.batch_size = batch_size

        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def index(
        self,
        corpus: pd.DataFrame,
        index_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:

        if batch_size is None:
            batch_size = self.batch_size

        self.docnos = corpus.docno

        # Create a FAISS IndexIDMap to store both vectors and their IDs
        self.faiss_index = faiss.index_factory(
            self.model.get_sentence_embedding_dimension(),
            "IDMap,Flat",
            self.faiss_metric,
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
                normalize_embeddings=(self.faiss_metric == faiss.METRIC_INNER_PRODUCT),
                batch_size=batch_size,
                prompt_name="document",
            ).astype(np.float32)

            # Add the embeddings to the index along with their IDs
            self.faiss_index.add_with_ids(embeddings, np.array(doc_ids))

        # Save the index if a path is provided
        if index_dir:
            self._save_index(index_dir)

        logger.info(f"Indexed {len(corpus)} documents.")

    def _save_index(self, index_dir: str) -> None:
        logger.info(f"Saving index and docnos to {index_dir}")
        os.makedirs(index_dir, exist_ok=True)

        faiss.write_index(
            self.faiss_index, os.path.join(index_dir, FAISS_INDEX_FILENAME)
        )

        self.docnos.to_csv(
            os.path.join(index_dir, DOCNOS_FILENAME),
            sep="\t",
            header=True,
            index=False,
        )

    def load_index(self, index_dir: str) -> None:
        self.faiss_index = faiss.read_index(
            os.path.join(index_dir, FAISS_INDEX_FILENAME)
        )
        self.docnos = pd.read_csv(
            os.path.join(index_dir, DOCNOS_FILENAME), sep="\t"
        ).docno

    def transform(self, queries: pd.DataFrame, k=1000) -> pd.DataFrame:

        if self.faiss_index is None:
            raise ValueError(
                "Index is not initialized - call self.index() or self.load_index() to load an index before trying to retrieve."
            )

        q_embs = self.model.encode(
            queries["query"],
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt_name="query",
        ).astype(np.float32)

        scores, idxs = self.faiss_index.search(q_embs, k=min(k, len(self.docnos)))

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
    dataset = ir_datasets.load(DATASET_TO_IRDS_NAME[dataset_name])

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
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    retriever_name: str,
    retriever: Retriever,
    dataset_name: str,
    neg_retriever_name: str,
    subsample_top_k: str,
    load_index: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    subsample_method_name = get_subsample_method_name(
        neg_retriever_name, subsample_top_k
    )
    index_dir = os.path.join(
        "index",
        dataset_name,
        subsample_method_name,
        retriever_name,
    )

    if load_index:
        retriever.load_index(index_dir=index_dir)
    else:
        retriever.index(corpus, index_dir=index_dir)

    retrieval_output = retriever.transform(queries, k=1000)

    save_results_dir = os.path.join(
        "results",
        dataset_name,
        subsample_method_name,
        retriever_name,
    )

    os.makedirs(save_results_dir, exist_ok=True)
    retrieval_output.to_csv(
        os.path.join(save_results_dir, RETRIEVER_OUTPUT_FILENAME),
        sep="\t",
        index=False,
        header=True,
    )

    if not pt.started():
        pt.init()

    eval_metrics = pt.Experiment(
        [retrieval_output],
        queries,
        qrels,
        eval_metrics=PT_METRICS,
        names=[retriever_name],
    )

    # add some extra info:
    _, dataset, subsample_method, retriever_name = save_results_dir.split("/")
    eval_metrics["dataset"] = dataset_name
    eval_metrics["neg_retriever_name"] = neg_retriever_name
    eval_metrics["subsample_top_k"] = subsample_top_k
    eval_metrics["retriever"] = retriever_name
    eval_metrics["num_docs"] = len(corpus)

    # drop name col, move new cols to front
    cols = eval_metrics.columns.to_list()
    new_cols = cols[-5:] + cols[1:-5]
    eval_metrics = eval_metrics[new_cols]

    eval_metrics.to_csv(
        os.path.join(save_results_dir, RESULTS_FILENAME),
        sep="\t",
        index=False,
        header=True,
    )

    return eval_metrics, retrieval_output


def top_k_docnos(retrieval_results: pd.DataFrame, top_k: int) -> Set[str | int]:
    return set(
        retrieval_results.groupby(by="qid")
        .apply(lambda x: x.nlargest(top_k, "score"), include_groups=False)
        .reset_index(drop=True)
        .docno
    )


def subsample_corpus(corpus: pd.DataFrame, keep_docnos: Set[str | int]) -> pd.DataFrame:
    return corpus[corpus["docno"].isin(keep_docnos)]


def get_retriever_output_path(
    dataset: str, subsample_method: str, retriever: str
) -> str:
    return os.path.join(
        "results",
        dataset,
        subsample_method,
        retriever,
        RETRIEVER_OUTPUT_FILENAME,
    )


def get_results_path(dataset: str, subsample_method: str, retriever: str) -> str:
    return os.path.join(
        "results",
        dataset,
        subsample_method,
        retriever,
        RESULTS_FILENAME,
    )


def get_subsample_method_name(neg_retriever_name: str, subsample_top_k: int) -> str:
    if neg_retriever_name == "full":
        return "full"
    else:
        return f"{neg_retriever_name}_top_{subsample_top_k}"


def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def full_experiment(
    retrievers: Dict[str, Retriever],
    neg_retriever_names: List[str],
    datasets: List[str],
    subsample_top_ks: List[int],
    load_index: bool = False,
):
    neg_retrievers: Dict[str, Retriever] = {
        name: retrievers[name] for name in neg_retriever_names
    }

    for dataset_name in datasets:
        logger.info(f"Dataset: {dataset_name}")
        corpus, queries, qrels = load_dataset(dataset_name)

        logger.info("Doing full retrieval")
        for retriever_name, retriever in retrievers.items():
            # do full retrieval
            eval_metrics, subsampled_retrieval_results = generate_results(
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                retriever=retriever,
                retriever_name=retriever_name,
                dataset_name=dataset_name,
                neg_retriever_name="full",
                subsample_top_k=0,
                load_index=load_index,
            )
            logger.info(f"Finished {retriever_name}")

        for neg_retriever_name, neg_retriever in neg_retrievers.items():
            full_retrieval_output = pd.read_csv(
                get_retriever_output_path(
                    dataset=dataset_name,
                    subsample_method="full",
                    retriever=neg_retriever_name,
                ),
                sep="\t",
            )

            for subsample_top_k in subsample_top_ks:
                subsample_method_name = get_subsample_method_name(
                    neg_retriever_name, subsample_top_k
                )

                keep_docnos = top_k_docnos(
                    full_retrieval_output, top_k=subsample_top_k
                ).union(set(qrels.docno))

                subsampled_corpus = subsample_corpus(corpus, keep_docnos)

                logger.info(
                    f"Doing retrieval on {dataset_name}/{subsample_method_name}"
                )

                for retriever_name, retriever in retrievers.items():
                    eval_metrics, subsampled_retrieval_results = generate_results(
                        corpus=subsampled_corpus,
                        queries=queries,
                        qrels=qrels,
                        retriever=retriever,
                        dataset_name=dataset_name,
                        retriever_name=retriever_name,
                        neg_retriever_name=neg_retriever_name,
                        subsample_top_k=subsample_top_k,
                        load_index=load_index,
                    )
                    logger.info(
                        f"Finished {subsample_method_name} with {retriever_name}"
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset to evaluate on")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    retrievers = {
        "snowflake-arctic-embed-m-long" : DenseRetriever(
            model_name_or_path="Snowflake/snowflake-arctic-embed-m-long",
            device="cuda",
            batch_size=256,
            max_seq_length=512,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        ),
        "gte-base-en-v1.5": DenseRetriever(
            model_name_or_path="Alibaba-NLP/gte-base-en-v1.5",
            device="cuda",
            batch_size=256,
            max_seq_length=512,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        ),
        "all-mpnet-base-v2": DenseRetriever(
            model_name_or_path="sentence-transformers/all-mpnet-base-v2",
            device="cuda",
            batch_size=256,
            max_seq_length=512,
            model_kwargs={"torch_dtype": torch.float16},
        ),
        "bge-base-en-v1.5": DenseRetriever(
            model_name_or_path="BAAI/bge-base-en-v1.5",
            device="cuda",
            batch_size=256,
            max_seq_length=512,
            model_kwargs={"torch_dtype": torch.float16},
        ),
        "e5-base-v2": DenseRetriever(
            model_name_or_path="intfloat/e5-base-v2",
            device="cuda",
            batch_size=256,
            max_seq_length=512,
            model_kwargs={"torch_dtype": torch.float16},
            prompts={"query": "query: ", "document": "passage: "},
        ),
        "jina-embeddings-v2-base-en-flash": DenseRetriever(
            model_name_or_path="jinaai/jina-embeddings-v2-base-en-flash",
            device="cuda",
            batch_size=256,
            max_seq_length=512,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        ),
        "BM25s": BM25s(lang="en", stemmer=Stemmer("english")),
    }

    neg_retriever_names = list(retrievers.keys())

    datasets = [args.dataset]

    subsample_top_ks = [1, 10, 50, 100, 250, 500, 750, 1000]

    full_experiment(
        retrievers=retrievers,
        neg_retriever_names=neg_retriever_names,
        datasets=datasets,
        subsample_top_ks=subsample_top_ks,
        load_index=False,
    )

    # dr = DenseRetriever(
    #     model_name_or_path="jinaai/jina-embeddings-v2-base-en-flash",
    #     device="cuda",
    #     max_seq_length=512,
    #     trust_remote_code=True,
    #     model_kwargs={"torch_dtype": torch.float16},
    # )

    # corpus, queries, qrels = load_dataset("beir/nq")

    # print("Indexing")
    # dr.index(
    #     corpus,
    #     index_dir="index/nq/full_corpus/jina-embeddings-v2-base-en-flash",
    #     batch_size=8192,
    # )

    # print("Loading Index")
    # dr.load_index(index_dir="index/nq/full_corpus/jina-embeddings-v2-base-en-flash")

    # results = dr.transform(queries, k=1000)

    # pt.init()

    # experiment = pt.Experiment(
    #     [dr],
    #     queries,
    #     qrels,
    #     eval_metrics=[nDCG @ 10, R @ 1000],
    #     names=["Dense Retriever"],
    # )
