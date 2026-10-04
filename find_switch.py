with open(r'c:\Users\Ratan Singh\AppData\Local\Claude-3p\local-agent-mode-sessions\d9198495\00000000\76289000\outputs\lightning_nowcast\index.html', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if 'switchTab' in line:
            print(f"Line {idx}: {line.strip()[:80].encode('ascii', 'ignore').decode('ascii')}")
