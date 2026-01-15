#!/usr/bin/env python3

import argparse

from lib.semantic_search import chunk_text, embed_query_text, embed_text, search, semantic_chunking, verify_embeddings, verify_model, embed_chunks

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model is setup")

    embed_parser = subparsers.add_parser("embed_text", help="Embed the input text")
    embed_parser.add_argument("text", type=str, help="Text to be embedded by the LLM") 

    verify_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings for movie dataset")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed the query")
    embedquery_parser.add_argument("query", type=str, help="Query to be embedded")

    embed_search = subparsers.add_parser("search", help="")
    embed_search.add_argument("query", type=str, help="Search query")
    embed_search.add_argument("--limit", type=int, default=5, help="Set the limit of documents to return along with their scores")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="String to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size to break up the string into")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Add overlap to ensure that context is preserved")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text but retain context")
    semantic_chunk_parser.add_argument("text", type=str, help="String to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max chunk size with a default of 4")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Optional overlap to ensure additional context is preserved")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Chunk and embed documents")

    search_chunk_parser = subparsers.add_parser("search_chunked", help="Used to search embedded chunks")
    search_chunk_parser.add_argument("query", type=str, help="Query string to search the chunks")
    search_chunk_parser.add_argument("--limit", type=int, default=10, help="Limit the results with an optional parameted, default is 10")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunking(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
