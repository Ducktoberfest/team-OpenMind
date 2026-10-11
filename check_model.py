import requests, ai_agent

r = requests.post(f"{ai_agent.OLLAMA_URL}/api/chat", json={
    "model": ai_agent.MODEL, "stream": False, "think": False,
    "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
    "options": {"num_predict": 1024}}, timeout=600)
d = r.json()
print("content:", repr(d["message"].get("content")))
print("thinking:", repr(d["message"].get("thinking", ""))[:200])
print("done_reason:", d.get("done_reason"))