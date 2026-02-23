import argparse

from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command



def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate multi-document summary"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of documents to summarize",
    )

    citation_parser = subparsers.add_parser(
        "citations", help="Search query to recieve a summary with citations"
    )
    citation_parser.add_argument("query", type=str, help="Query to be searched, summarized, and cited")
    citation_parser.add_argument("--limit", type=int, default=5, help="Optional maximum number of documents to summarize")

    question_parser = subparsers.add_parser(
        "question", help="Generate an answer for a user with the LLM"
    )
    question_parser.add_argument("question", type=str, help="The user's question for the LLM to be answered")
    question_parser.add_argument("--limit", type=int, default=5, help="Optional maximum number of documents to return")

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
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Summary:")
            print(result["summary"])
        case "citations":
            result = citations_command(args.query, args.limit) 
            print("Search Results:")
            for document in result["search_results"]:
                print(f"    -{document["title"]}")
            print()
            print("LLM Answer:")
            print(result['cited_summary'])
        case "question":
            result = question_command(args.question, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"    -{document["title"]}")
            print()
            print("Answer:")
            print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
