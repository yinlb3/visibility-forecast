with open(r'd:\Project\vis\draw.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

long_lines = []
for i, line in enumerate(lines, 1):
    if len(line.rstrip()) > 80:
        long_lines.append((i, len(line.rstrip()), line.rstrip()[:70]))

if long_lines:
    print(f'Found {len(long_lines)} lines exceeding 80 characters:')
    for line_num, length, preview in long_lines[:50]:
        print(f'  Line {line_num} ({length} chars): {preview}...')
else:
    print('No lines exceed 80 characters.')
