import json
import random
import os
import time
import typing_extensions
from typing import List, Dict, Any

# Try importing the Google Gen AI library
try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import content
except ImportError:
    print("Error: 'google-generativeai' library not found. Please install it using: pip install google-generativeai")
    exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'data/sections.json'
OUTPUT_FILE = 'data/raft_dataset.jsonl'
NUM_DISTRACTORS = 4
MAX_ORACLES = 5 
SAMPLE_SIZE = 100

# !!! PASTE YOUR GEMINI API KEY HERE !!!
GEMINI_API_KEY = "AIzaSyBmfofqNAwvzCu3OqsNnRvHEcbVTIOWLzQ" 

# Model Configuration
MODEL_NAME = "gemini-2.5-flash" # Cost-effective and fast
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# ==========================================
# DATA LOADING & PROCESSING
# ==========================================

def load_data(filepath: str) -> List[Dict]:
    """Loads data. If it's a dict (sections.json), converts to a list of dicts with 'id' injected."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        processed_list = []
        for key, value in data.items():
            item = value.copy()
            item['id'] = key
            processed_list.append(item)
        return processed_list
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unknown JSON format")

def get_distractors(all_sections: List[Dict], oracle_ids: List[str], k: int = 4) -> List[str]:
    """Selects k random sections as distractors, excluding the oracle sections."""
    pool = [s for s in all_sections if s['id'] not in oracle_ids]
    if len(pool) < k:
        selected = pool
    else:
        selected = random.sample(pool, k)
    return [s['content'] for s in selected]

# ==========================================
# GEMINI API INTERACTION
# ==========================================

def generate_qa_pair(oracle_contents: List[str]) -> Dict[str, str]:
    """
    Generates a Question, Chain of Thought, and Answer using Gemini.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please edit the script to add your key.")

    combined_text = "\n\n--- NEXT SECTION ---\n\n".join(oracle_contents)
    
    # Define the schema for structured JSON output
    # Note: Gemini 1.5 supports JSON mode natively
    
    prompt = f"""You are a teacher preparing a dataset for Retrieval Augmented Generation (RAG).
Your task is to generate a high-quality question, a chain-of-thought reasoning process, and a final answer based ONLY on the provided text sections.

Input Text:
{combined_text}

Instructions:
1. Generate a question that requires understanding the provided text. If multiple sections are provided, try to synthesize information from them.
2. Provide a "Chain of Thought" (CoT) that explains how the answer is derived from the text. Explicitly cite or refer to the text content.
3. Provide the final concise Answer.

Output Format (JSON):
{{
  "question": "The generated question",
  "cot": "The step-by-step reasoning...",
  "answer": "The final answer"
}}
"""

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG
        )
        
        response = model.generate_content(prompt)
        
        # Parse JSON response
        result = json.loads(response.text)
        return result

    except Exception as e:
        print(f"Error generating QA with Gemini: {e}")
        # Return fallback/empty to allow script to continue (or raise if you prefer strict failure)
        return {
            "question": "Error generating question",
            "cot": "Error generating CoT",
            "answer": "Error generating answer"
        }

# ==========================================
# RAFT ENTRY CREATION
# ==========================================

def create_raft_entry(all_sections: List[Dict]) -> Dict:
    """Creates a single RAFT dataset entry in Llama 3.1 'messages' format."""
    
    # 1. Select Oracles
    num_oracles = random.randint(1, MAX_ORACLES)
    oracles = random.sample(all_sections, num_oracles)
    oracle_ids = [o['id'] for o in oracles]
    oracle_contents = [o['content'] for o in oracles]
    
    # 2. Generate Q&A (Using Gemini)
    # Add a small sleep to avoid hitting rate limits too hard if running sequentially
    time.sleep(1) 
    qa = generate_qa_pair(oracle_contents)
    
    # 3. Get Distractors
    distractors = get_distractors(all_sections, oracle_ids, k=5 - MAX_ORACLES)
    
    # 4. Combine Context
    context_docs = oracle_contents + distractors
    random.shuffle(context_docs)
    
    # 5. Format Context
    formatted_context = ""
    for i, doc in enumerate(context_docs):
        formatted_context += f"<DOCUMENT id='{i}'>\n{doc}\n</DOCUMENT>\n\n"
        
    # 6. Construct Messages
    user_content = f"""Given the following documents, answer the question. 
Provide a Chain of Thought reasoning before your final answer.

{formatted_context}

Question: {qa['question']}"""

    assistant_content = f"Chain of Thought: {qa['cot']}\n\nAnswer: {qa['answer']}"

    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based strictly on the provided documents. You must first provide your reasoning (Chain of Thought) and then the final answer."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "oracle_ids": oracle_ids,
            "num_distractors": 5 - MAX_ORACLES
        }
    }

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # Configure API
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        print("WARNING: GEMINI_API_KEY is empty. The script will fail when it tries to generate content.")

    print(f"Loading data from {INPUT_FILE}...")
    sections = load_data(INPUT_FILE)
    print(f"Loaded {len(sections)} sections.")
    
    output_data = []
    
    print(f"Generating {SAMPLE_SIZE} dataset entries using {MODEL_NAME}...")
    
    for i in range(SAMPLE_SIZE):
        print(f"Processing entry {i + 1}/{SAMPLE_SIZE}...")
        try:
            entry = create_raft_entry(sections)
            output_data.append(entry)
        except Exception as e:
            print(f"Failed to create entry {i}: {e}")
            continue

    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    main()
