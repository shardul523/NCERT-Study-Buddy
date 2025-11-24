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
paragraphs = para6 + para7 + para8

# 3. Merge Sections (Dictionary merging)
sections = {}
sections.update(section6)
sections.update(section7)
sections.update(section8)

# final_paragraphs = []
final_sections = {}

for id, data in sections.items():
    section_content = ''
    for i, chunk in enumerate(paragraphs):
        if chunk['parent_id'] == id:
            section_content += f'{chunk['content']} \n'
        paragraphs[i]['metadata'] = data
    # sections[id]['content'] = section_content
    final_sections[id] = {
        'content': section_content,
        'metadata': data
    }

# 4. Save Data
with open('data/paragraphs.json', 'w', encoding='utf-8') as file:
    json.dump(paragraphs, file, indent=4)

with open('data/sections.json', 'w', encoding='utf-8') as file:
    json.dump(final_sections, file, indent=4)

print("Data successfully loaded, merged, and saved to 'data/paragraphs.json' and 'data/sections.json'")