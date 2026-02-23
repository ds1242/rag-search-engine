import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embed = subparser.add_parser("verify_image_embedding", help="Verify image embeddings")
    verify_image_embed.add_argument("image", type=str, help="path to image to embed")


    image_search_parser = subparser.add_parser("image_search", help="search by an image and text")
    image_search_parser.add_argument("image_path", type=str, help="path to image to be searched")

    args = parser.parse_args()


    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)

        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"     {res['description'][:100]}...")


        case _:
            parser.print_help()









if __name__ == "__main__":
    main()
