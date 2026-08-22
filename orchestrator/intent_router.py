import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = """You are an intent router for a farm advisory system.
Given a farmer's question (in Telugu), decide which of these topics it relates to.
A question can relate to MORE THAN ONE topic at the same time — check each one independently, don't just pick the single best match.

- disease: crop disease, pests, leaf spots, plant health, spraying medicine on crops
- weather: rain, forecast, temperature, climate
- price: market price, selling, buying, mandi rates
- soil: soil health, pH, nutrients, fertilizer

Respond with ONLY a JSON object with these exact keys, no other text:
{"wants_disease": true/false, "wants_weather": true/false, "wants_price": true/false, "wants_soil": true/false}

Examples:

Question: వర్షం వస్తే మందు కొట్టాలా? (Should I spray if rain is coming?)
JSON: {"wants_disease": true, "wants_weather": true, "wants_price": false, "wants_soil": false}

Question: నా పంట అమ్మాలా ఆపాలా, ధర గురించి చెప్పు వాతావరణం కూడా చూసి (Should I sell or hold, considering price and weather?)
JSON: {"wants_disease": false, "wants_weather": true, "wants_price": true, "wants_soil": false}

Question: ఈ రోజు టమాటా ధర ఎంత? (What's today's tomato price?)
JSON: {"wants_disease": false, "wants_weather": false, "wants_price": true, "wants_soil": false}

Question: నా పొలంలో మట్టి బాగుందా? (Is my field's soil good?)
JSON: {"wants_disease": false, "wants_weather": false, "wants_price": false, "wants_soil": true}

Question: నమస్కారం (Hello)
JSON: {"wants_disease": false, "wants_weather": false, "wants_price": false, "wants_soil": false}

If the question doesn't relate to any topic, return all false.
"""

def route_intent(question: str) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nJSON:"

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    })
    response.raise_for_status()
    raw_output = response.json()["response"].strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "wants_disease": False,
            "wants_weather": False,
            "wants_price": False,
            "wants_soil": False,
            "_error": f"Could not parse model output: {raw_output}",
        }

if __name__ == "__main__":
    test_question = "నా టమాటా ఆకులపై మచ్చలు ఉన్నాయి, ఇది ఏ వ్యాధి?"
    result = route_intent(test_question)
    print(json.dumps(result, indent=2, ensure_ascii=False))
