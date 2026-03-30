import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.core.qa_chain import QAChain

def main():
    parser = argparse.ArgumentParser(description="WeightLoss RAG Command Line Interface")
    parser.add_argument(
        "query", 
        type=str, 
        help="The medical question to search for."
    )
    
    args = parser.parse_args()
    user_query = args.query
    
    print("\n" + "="*80)
    print("🧠 WeightLoss RAG - Processing Query...")
    print("="*80)
    print(f"QUESTION: {user_query}")
    print("-" * 80)
    
    qa_chain = QAChain()
    
    try:
        # Run the full RAG pipeline (Parse -> Retrieve -> Format -> Generate)
        answer, strategy = qa_chain.query(user_query)
        
        print("\n🤖 ANSWER:\n")
        print(answer)
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate answer. Details: {e}")
        
if __name__ == "__main__":
    main()
