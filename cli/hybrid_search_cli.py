import argparse

from lib.hybrid_search import normalize_score, weighted_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="command that tests normalizing scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="the list of scores being normalized")

    weighted_search_parser = subparser.add_parser("weighted-search", help="conduct a weighted search using optional commands --alpha an --limit")
    weighted_search_parser.add_argument("query", type=str, help="the query provided to be searched")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="The alpha value to be added which can adjust how the score is calculated, default = 0.5")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit how many results returned, default = 5")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_score(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
