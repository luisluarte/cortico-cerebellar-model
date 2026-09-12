import re
def extract_field(field, text):
    pattern = re.compile(rf'{field}\s*=\s*\{{', re.IGNORECASE)
    match = pattern.search(text)
    if not match: return None
    start = match.end(); brace_count = 1; i = start
    while i < len(text) and brace_count > 0:
        if text[i] == '{': brace_count += 1
        elif text[i] == '}': brace_count -= 1
        i += 1
    return text[start:i-1].replace('\n', ' ').replace('  ', ' ').strip()

with open(r'c:\Users\DCCS5\Documents\GitHub\cortico-cerebellar-model\manuscript\cerebellar_lib.bib', 'r', encoding='utf-8') as f: content = f.read()
entries = re.split(r'@\w+\{', content)[1:]
for i, entry in enumerate(entries):
    title = (extract_field('title', entry) or 'Unknown Title').replace('{','').replace('}','')
    filepath = extract_field('file', entry) or 'No local PDF'
    abstract = extract_field('abstract', entry) or 'No abstract available.'
    summary = abstract
    if summary != 'No abstract available.':
        sentences = [s.strip() for s in summary.split('. ') if s.strip()]
        if len(sentences) > 2: summary = '. '.join(sentences[:2]) + '.'
        elif len(sentences) > 0: summary = sentences[0] + '.'
        if len(summary) > 400: summary = summary[:397] + '...'
    if summary == 'No abstract available.': summary = f'Study regarding {title}.'
    print(f'Digesting Paper {i+1}/{len(entries)}')
    print(f'- Title: {title}')
    print(f'- PDF Location: {filepath}')
    print(f'- Summary: {summary}\n')

