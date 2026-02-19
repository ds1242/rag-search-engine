import argparse

from lib.augmented_generation import (
    rag_command,
    summarize_command,
    )


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summary the movies returned")

    summarize_parser.add_argument("query", type=str, help="Search command for RAG query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit number of results rteturned (default=5)")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            results = summarize_command(args.query, args.limit)
            print("Search Results")
            for document in results['search_results']:
                print(f"    - {document['title']}")
            print()
            print("RAG Response")
            print(results['summaries'])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
