#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            base_dir = Path("data")

            file_to_open = base_dir / "movies.json"

            print(file_to_open)

            with open(file_to_open, 'r') as f:
                data = json.load(f)
            
                
            movie_list = []
            for movie in data['movies']:
                if args.query in movie['title']:
                    movie_list.append(movie)

            for movie in movie_list[:5]:
                print(f"{movie['id']}. {movie['title']}")

            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
