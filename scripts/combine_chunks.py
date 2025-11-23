import json

# Define file paths
para6 = 'data/paragraphs6.json'
para7 = 'data/paragraphs7.json'
para8 = 'data/paragraphs8.json'

section6 = 'data/sections6.json'
section7 = 'data/sections7.json'
section8 = 'data/sections8.json'

def load_json(path: str):
    """Loads a JSON file from the given path."""
    with open(path, 'r', encoding='utf-8') as file:
        obj = json.load(file)
    return obj

# 1. Load Data
para6 = load_json(para6)
para7 = load_json(para7)
para8 = load_json(para8)

section6 = load_json(section6)
section7 = load_json(section7)
section8 = load_json(section8)

# 2. Merge Paragraphs (List concatenation)
# Assumes paragraphs are lists/arrays and need to be appended.
paragraphs = para6 + para7 + para8

# 3. Merge Sections (Dictionary merging)
# This uses the dictionary update() method.
# The order section6, section7, section8 ensures that values from
# section8 will overwrite any matching keys from section7 or section6, 
# and section7 will overwrite section6, matching the logic in the original code.
sections = {}
sections.update(section6)
sections.update(section7)
sections.update(section8)

# Optional Python 3.9+ way:
# sections = section6 | section7 | section8 

# 4. Save Data
# It's good practice to use an explicit encoding like 'utf-8'
with open('data/paragraphs.json', 'w', encoding='utf-8') as file:
    # Use indent for readability in the output file
    json.dump(paragraphs, file, indent=4)

with open('data/sections.json', 'w', encoding='utf-8') as file:
    # Use indent for readability in the output file
    json.dump(sections, file, indent=4)

print("Data successfully loaded, merged, and saved to 'data/paragraphs.json' and 'data/sections.json'")