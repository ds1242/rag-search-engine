#!/usr/bin/env python3

import argparse

from lib.keyword_search import InvertedIndex, search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build cache")

    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            try:
                index.load()
            except Exception as e:
                print(f"Error: {e}")
                return

            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            index.build()
            index.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
