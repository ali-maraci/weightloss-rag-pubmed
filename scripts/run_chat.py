import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.chat_engine import ChatEngine

def main():
    print("=" * 80)
    print("🧠 WeightLoss RAG Conversational Engine")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'clear' to wipe the conversational memory.")
    print("=" * 80 + "\n")
    
    chat_engine = ChatEngine()
    
    while True:
        try:
            user_input = input("\n👤 YOU: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            if user_input.lower() == 'clear':
                chat_engine.clear_history()
                print("--- Memory Cleared ---")
                continue
                
            print("\n🤖 AURAQUERY:")
            
            # This triggers Reformulation -> RAG -> Response
            answer = chat_engine.chat(user_input)
            
            print(f"{answer}\n")
            print("-" * 80)
            
        except KeyboardInterrupt:
            print("\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()
