import time
from ranking import rank
from reranker import rerank

def main():
    start = time.time()
    
    print("\n=== Stage 1/2: ranking (semantic + capability + structured + behavioral) ===")
    rank.main()

    print("\n=== Stage 2/2: deep-evidence rerank + submission.csv ===")
    rerank.main()

    elapsed = time.time() - start
    print(f"\nPipeline complete in {elapsed:.1f}s")
    
    if elapsed > 300:
        print("WARNING: Pipeline exceeded the 5-minute (300s) constraint!")

if __name__ == "__main__":
    main()