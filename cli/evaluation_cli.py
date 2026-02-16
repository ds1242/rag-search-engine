import argparse
import json

from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open('data/golden_dataset.json', 'r') as f:
        data = json.load(f)
    
    results = []
    i = 0
    for case in data['test_cases']: 
        curr_res = {}
        retrieved_titles = []
        relevant_titles = []
        result = rrf_search_command(query=case['query'], k=60, limit=limit)

        for res in result['results']:
            retrieved_titles.append(res['title'])

        for title in retrieved_titles:
            if title in case['relevant_docs']:
                relevant_titles.append(title)
        # print(case)
        curr_res = {
            'query': case['query'],
            'relevant': relevant_titles,
            'retrieved': retrieved_titles,
        }
        results.append(curr_res)
        i += 1


    for res in results:
        print(f"- Query: {res['query']}")
        print(f"    - Precision@{limit}: {len(res['relevant'])/ len(res['retrieved']):.4f}")
        print(f"    - Retrieved: {", ".join(res['retrieved'])}") 
        print(f"    - Relevent: {", ".join(res['relevant'])}")
        
if __name__ == "__main__":
    main()
