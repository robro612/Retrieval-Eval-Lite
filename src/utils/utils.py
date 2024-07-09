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
from pyterrier.measures import AP, R, nDCG
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DataTypes:
    # documents are [docno, text]
    Document = namedtuple("Document", ["docno", "text"])
    # queries are [qid, query]
    Query = namedtuple("Query", ["qid", "query"])
    # qrels are [qid, docno, label]
    Qrel = namedtuple("Qrel", ["qid", "docno", "label"])
    # retrieval results are a pd.DataFrame with [qid, docno, score, rank]
    Result = namedtuple("Result", ["qid", "docno", "score", "rank"])


class Retriever(pt.Transformer, ABC):

    @abstractmethod
    def index(self, corpus: pd.DataFrame, index_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def load_index(self, index_path: str) -> None:
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
        index_path: Optional[str] = None,
        load_index: bool = False,
    ):
        self.corpus = corpus
        self.lang = lang
        self.stemmer = stemmer
        self.index_path = index_path

        self.retriever = None

    def index(self, corpus: pd.DataFrame, index_path: Optional[str] = None) -> None:

        self.retriever = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(
            corpus["text"].to_list(), stopwords=self.lang, stemmer=self.stemmer
        )
        self.retriever.index(corpus_tokens)

        if self.index_path is not None:
            print(f"Saving index to {self.index_path}")
            self.retriever.save(self.index_path)

    def load_index(self, index_path: str) -> None:
        self.retriever = bm25s.BM25.load(index_path)

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        if self.retriever is None:
            raise ValueError(
                "Retriever is not initialized - call self.index() or self.load_index() to load an index before trying to retrieve."
            )

        query_token_ids = bm25s.tokenize(
            queries["query"].to_list(), stopwords=self.lang, stemmer=self.stemmer
        )
        docnos, scores = self.retriever.retrieve(
            query_token_ids, corpus=self.corpus["docno"].to_list(), k=1000
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
        metric = faiss.METRIC_INNER_PRODUCT,
        **sentence_transformer_kwargs: Dict[str, Any],
    ):
        self.metric = metric
        self.model = SentenceTransformer(**sentence_transformer_kwargs)

    def index(
        self,
        corpus: pd.DataFrame,
        index_path: Optional[str] = None,
        batch_size: int = 2048,
    ) -> None:
        # Create a FAISS IndexIDMap to store both vectors and their IDs
        dimension = self.model.get_sentence_embedding_dimension()

        metric = self.metric
        index = faiss.index_factory(dimension, 'IDMap,Flat', self.metric)

        # Process the corpus in batches
        for i in tqdm(range(0, len(corpus), batch_size), desc="Indexing"):
            batch = corpus.iloc[i : i + batch_size]

            # Extract text and IDs from the batch
            texts = batch["text"].tolist()
            doc_ids = batch.index.tolist()

            # Encode the texts into embeddings
            embeddings = self.model.encode(
                texts, show_progress_bar=False,
            )

            # Add the embeddings to the index along with their IDs
            index.add_with_ids(
                embeddings, torch.tensor(doc_ids),
            )

        # Save the index if a path is provided
        if index_path:
            faiss.write_index(index, index_path)

        # Store the index in the class instance
        self.faiss_index = index

        # Store the mapping of FAISS IDs to original document IDs
        self.id_mapping = dict(enumerate(corpus.index))

        print(f"Indexed {len(corpus)} documents.")
