#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, search_command, tf_command, idf_command, tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser = subparsers.add_parser("tf", help="Get a term frequency using a doc id and term")
    search_parser.add_argument("doc_id", type=int, help="Document ID to search for term frequency")
    search_parser.add_argument("term", type=str, help="Term to check frequency")

    search_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a term")
    search_parser.add_argument("idf_term", help="Term to check idf value")

    search_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score for a given a document id and term")

    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)

            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            print(f"Searching {args.doc_id} for {args.term}")
            count = tf_command(args.doc_id, args.term)
            print(f"{count}")
        case "idf":
            idf_val = idf_command(args.idf_term)
            print(f"Inverse document frequency of '{args.idf_term}' : {idf_val:.2f}")
        case "tfidf":
            tfidf_val = tfidf_command(args.doc_id, args.term)
            print(tfidf_val)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
