import argparse
import logging

from utils.text_utils import TextEncoder, VectorStore, process_pdf_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    args = parse_args()
    logging.info(f"Processing PDF folder: {args.pdf_folder}")
    logging.info(f"Query: {args.query}")
    logging.info(f"Number of results: {args.num_results}")

    encoder = TextEncoder()
    texts = process_pdf_folder(args.pdf_folder)
    if not texts:
        logging.warning("No text found in PDF files!")
        return

    embeddings = encoder.encode(texts)
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_texts(texts, embeddings)

    logging.info(f"\nSearching for: {args.query}")
    query_embedding = encoder.encode(args.query)
    results = vector_store.search(query_embedding, k=args.num_results)

    print("\nResults:")
    print("-" * 80)
    for i, (text, score) in enumerate(results, 1):
        print(f"\n{i}. Distance Score: {score:.4f}")
        print(f"{text[:500]}...")  # Show first 300 characters
        print("-" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-modal search engine")
    parser.add_argument(
        "--pdf_folder",
        type=str,
        required=True,
        help="Path to the folder containing PDFs",
    )
    parser.add_argument("--query", type=str, required=True, help="Query to search for")
    parser.add_argument(
        "--num_results", type=int, default=3, help="Number of results to return"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
