import math
from collections import Counter, defaultdict


class CustomBM25Engine:

    def __init__(self, corpus, k1=1.5, b=0.75):

        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        # Tokenize documents into simple lower-case words
        self.docs = [
            doc.lower().split()
            for doc in corpus
        ]

        self.doc_lengths = [
            len(doc)
            for doc in self.docs
        ]

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.corpus_size
            if self.corpus_size > 0
            else 0
        )

        # Build term frequencies for every document
        self.doc_term_frequencies = [
            Counter(doc)
            for doc in self.docs
        ]

        # Inverted index:
        # term -> number of documents containing the term
        self.inverted_index = defaultdict(int)

        for doc in self.docs:
            unique_terms = set(doc)

            for term in unique_terms:
                self.inverted_index[term] += 1


    def _calculate_idf(self, term):

        df = self.inverted_index.get(term, 0)

        # Smooth Inverse Document Frequency formula
        return math.log(
            (self.corpus_size - df + 0.5)
            / (df + 0.5)
            + 1.0
        )


    def score_document(self, query, doc_index):

        query_terms = query.lower().split()

        doc_len = self.doc_lengths[doc_index]

        tf_dict = self.doc_term_frequencies[doc_index]

        score = 0.0

        for term in query_terms:

            if term not in tf_dict:
                continue

            # Term Frequency
            tf = tf_dict[term]

            # Inverse Document Frequency
            idf = self._calculate_idf(term)

            # Standard BM25 Term Frequency scaling
            numerator = tf * (self.k1 + 1)

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (doc_len / self.avg_doc_length)
                )
            )

            score += (
                idf
                * (numerator / denominator)
            )

        return score


    def rank_documents(self, query, top_n=2):

        scores = []

        for i in range(self.corpus_size):

            score = self.score_document(
                query,
                i
            )

            scores.append(
                (i, score)
            )

        # Sort documents by BM25 score
        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Return Top-K documents
        return scores[:top_n]