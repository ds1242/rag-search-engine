import argparse
from lib.describe_image import describe_command

def main():
    parser = argparse.ArgumentParser(description="Describe images CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    image_parser = subparser.add_parser(
        "--image", type=str, help="Path to input image"
    )
    image_parse = subparser.add_parser("--query", type=str, help="a text query to rewrite based on the image")

    args = parser.parse_args()
    
    match args.command:
        case "image":
            describe_command(args.image, args.query)


if __name__ == "__main__":
    main()
