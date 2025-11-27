import json
import time
import os
import google.generativeai as genai
from typing import List, Dict

# --- CONFIGURATION ---
API_KEY = "AIzaSyBgxwH57Nzn6ZwBenTpYdxayRwgJ3k07kA"  # Replace with your actual key
FILE_PATH_A = "data/rag_self_ask.json"   # Your first file
FILE_PATH_B = "data/rag_self_ask_raft.json"   # Your second file
OUTPUT_FILE = "data/evaluation_results_self_ask_raft.json"
MODEL_NAME = "gemini-2.0-flash" # Flash is fast and good for evaluation

# Configure the SDK
genai.configure(api_key=API_KEY)

def load_data(filepath: str) -> List[Dict]:
    """Loads the JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return []

def get_judge_verdict(question: str, answer_a: str, answer_b: str) -> Dict:
    """
    Sends the pair to Gemini to pick a winner.
    Returns a dictionary with 'winner' and 'reasoning'.
    """
    
    # We use a system prompt to define the persona
    system_instruction = (
        """
You are an objective evaluator. Your task is to judge which of two candidate answers is better for a given question.

Inputs You Will Receive:

QUESTION – the original query.

ANSWER A – the first model’s answer.

ANSWER B – the second model’s answer.

Evaluation Criteria (in order of importance):

Correctness / Factual Accuracy – Does the answer align with known facts or logically valid reasoning?

Completeness – Does the answer address all parts of the question?

Relevance – Does the answer stay focused on the question without unnecessary information?

Clarity – Is the explanation easy to understand and well-structured?

Conciseness – Is the answer free of redundancy and unnecessary length (without losing important information)?

Important Instructions:

Judge only the content of the answers, not style preferences or verbosity alone.

Do not invent facts or fill in missing information. Base your judgment solely on the provided answers.

Avoid bias toward longer answers; choose the longer one only if it is genuinely more correct and complete.

If both answers are equally good or equally poor, you may output "TIE".

Do not rewrite or improve the answers. Only evaluate them.

Output Format:
Respond only in the following JSON format:

{
  "winner": "A" | "B" | "TIE",
  "reasoning": "A concise explanation (2–4 sentences) justifying your decision."
}

        """
    )

    # The specific prompt for this turn
    prompt = f"""
    Please evaluate the following:

    QUESTION:
    {question}

    ANSWER A:
    {answer_a}

    ANSWER B:
    {answer_b}

    """

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"}
    )

    try:
        response = model.generate_content(prompt)
        # Parse the JSON response text
        return json.loads(response.text)
    except Exception as e:
        print(f"Error calling API: {e}")
        print({"winner": "Error", "reasoning": str(e)})
        print('Retrying in 1 minute')
        time.sleep(60)
        response = model.generate_content(prompt)
        return json.loads(response.text)

def main():
    # 1. Load the files
    print("Loading files...")
    data_a = load_data(FILE_PATH_A)
    data_b = load_data(FILE_PATH_B)

    if len(data_a) != len(data_b):
        print("Warning: Files have different numbers of items. Comparing only up to the shortest list.")

    # 2. Iterate and Compare
    results = []
    
    # Zip allows us to iterate through both lists simultaneously
    for index, (item_a, item_b) in enumerate(zip(data_a, data_b)):
        
        # specific check to ensure we are comparing the same question
        # (Optional: depends on if your files are perfectly aligned)
        question = item_a.get("question")
        
        print(f"Evaluating pair #{index + 1}...")
        
        verdict = get_judge_verdict(
            question=question,
            answer_a=item_a.get("answer"),
            answer_b=item_b.get("answer")
        )
        
        # Structure the result
        result_entry = {
            "question": question,
            "answer_a": item_a.get("answer"),
            "answer_b": item_b.get("answer"),
            "winner": verdict.get("winner"),
            "reasoning": verdict.get("reasoning")
        }
        
        results.append(result_entry)
        
        # Sleep to avoid hitting rate limits (adjust based on your tier)
        time.sleep(2) 

    # 3. Save Results
    print(f"Finished. Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # 4. Quick Summary
    wins_a = sum(1 for r in results if r['winner'] == 'A')
    wins_b = sum(1 for r in results if r['winner'] == 'B')
    ties = sum(1 for r in results if r['winner'] == 'Tie')
    
    print("-" * 30)
    print("FINAL SCOREBOARD")
    print(f"Model A Wins: {wins_a}")
    print(f"Model B Wins: {wins_b}")
    print(f"Ties:         {ties}")
    print("-" * 30)

if __name__ == "__main__":
    main()