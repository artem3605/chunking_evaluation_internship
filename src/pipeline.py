from src.data_loader import get_corpus, get_questions_df
from src.embedding import generate_embeddings
from src.evaluation import Evaluation
from src.fixed_token_сhunker import FixedTokenChunker
from src.retriever import Retriever


def run_pipeline(config):
    """
    Run the entire pipeline for chunking evaluation.
    """
    corpus = get_corpus(config["corpus"])
    questions_df = get_questions_df(config["corpus"])

    chunker = FixedTokenChunker(chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
    chunks = chunker.split_text(corpus)
    embeddings = generate_embeddings(chunks)
    questions_df["embeddings"] = list(generate_embeddings(questions_df["question"].tolist()))

    retriever = Retriever(embeddings)
    top_chunks = [[chunks[i] for i in retriever.retrieve(query, config["retrieved_chunks"])] for query in
                  questions_df["embeddings"]]
    evaluation = Evaluation(corpus, questions_df, chunker)
    return evaluation.calc_metrics(top_chunks)


cfg = {
    "corpus": "wikitexts",
    "chunk_size": 200,
    "chunk_overlap": 0,
    "retrieved_chunks": 5,
}

if __name__ == "__main__":
    metrics = run_pipeline(cfg)
    print(metrics)
