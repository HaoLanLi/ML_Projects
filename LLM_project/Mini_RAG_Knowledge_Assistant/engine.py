import math
from collections import Counter, defaultdict

class BM25Engine:

    def __init__(self, chunks, k1=1.5, b=0.75):

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # Number of chunks
        self.corpus_size = len(chunks)

        # Tokenized content of each chunk
        self.docs = [
            chunk["content"].lower().split()
            for chunk in chunks
        ]

        # Length of each chunk
        self.doc_lengths = [
            len(doc)
            for doc in self.docs
        ]

        # Average chunk length
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.corpus_size
            if self.corpus_size > 0
            else 0
        )

        # Term frequency for each chunk
        self.doc_term_frequencies = [
            Counter(doc)
            for doc in self.docs
        ]

        # Inverted index
        self.inverted_index = defaultdict(int)

        for doc in self.docs:
            unique_terms = set(doc)

            for term in unique_terms:
                self.inverted_index[term] += 1


    def _calculate_idf(self, term):
        df = self.inverted_index.get(term, 0)

        idf = math.log((self.corpus_size-df+0.5)/(df+0.5)+1)

        return max(idf,0.0001)  # Ensure IDF is not negative or zero to avoid division by zero in scoring.


    def score_document(self, query, doc_index):

        query_terms = query.lower().split()

        doc_len = self.doc_lengths[doc_index]
        tf_dict = self.doc_term_frequencies[doc_index]

        score = 0.0

        for term in query_terms:

            if term not in tf_dict:
                continue

            tf = tf_dict[term]

            idf = self._calculate_idf(term)

            numerator = tf * (self.k1 + 1)

            denominator = (
                tf
                + self.k1
                * (
                    1 - self.b
                    + self.b
                    * (doc_len / self.avg_doc_length)
                )
            )

            score += idf * (numerator / denominator)

        return score


    def rank_documents(self, query, top_n=3):

        scores = []

        for i in range(self.corpus_size):

            score = self.score_document(query, i)

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        top_chunks = []
        for doc_index, score in scores[:top_n]:
            chunk_result = self.chunks[doc_index].copy()
            chunk_result["score"]=round(score, 3)
            top_chunks.append(chunk_result)
        return top_chunks