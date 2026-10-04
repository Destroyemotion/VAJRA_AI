with open(r'c:\Users\Ratan Singh\AppData\Local\Claude-3p\local-agent-mode-sessions\d9198495\00000000\76289000\outputs\lightning_nowcast\index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('Original line count:', len(lines))

# Lines 630 to 1289 (0-indexed 629 to 1289) are duplicate orphan blocks placed inside header.
# Let's inspect line 625-630 and line 1288-1293 before cutting.
print("Line 628-632:", [lines[i].strip() for i in range(627, 632)])
print("Line 1288-1292:", [lines[i].strip() for i in range(1287, 1292)])

# Keep lines 0 to 628, then insert </header>, then keep lines from 1290 onwards!
header_close = "  </header>\n"
new_lines = lines[:629] + [header_close] + lines[1290:]

print('New line count:', len(new_lines))

with open(r'c:\Users\Ratan Singh\AppData\Local\Claude-3p\local-agent-mode-sessions\d9198495\00000000\76289000\outputs\lightning_nowcast\index.html', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Successfully cleaned up index.html DOM structure!')
