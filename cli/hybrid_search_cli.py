import argparse

from lib.hybrid_search import normalize_score

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="command that tests normalizing scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="the list of scores being normalized")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_score(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
