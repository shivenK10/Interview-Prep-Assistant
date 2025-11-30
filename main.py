from pathlib import Path
from generation_pipeline import generate_answer
from logger import Logger

Path("Logs").mkdir(parents=True, exist_ok=True)
log = Logger("Main Application", log_file_needed=True, log_file="Logs/main.log", level="DEV")

def main():
    log.info("Chat started")
    print("\nInterview Prep Assistant")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                log.info("Chat ended")
                print("Goodbye!")
                break
            
            log.info(f"Question: {question}")
            result = generate_answer(question, top_k=5)
            log.info("Answer generated")
            
            print(f"\nAssistant: {result['answer']}\n")
            
        except KeyboardInterrupt:
            log.warning("Interrupted")
            print("\n\nGoodbye!")
            break
        except Exception as e:
            log.error(f"Error: {str(e)}")
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()
