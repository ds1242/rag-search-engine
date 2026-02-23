import argparse
from lib.describe_image import describe_command

def main():
    parser = argparse.ArgumentParser(description="Describe images CLI")

    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--query", type=str, required=True, help="a text query to rewrite based on the image")

    args = parser.parse_args()
    
    describe_command(args.image, args.query)


if __name__ == "__main__":
    main()
