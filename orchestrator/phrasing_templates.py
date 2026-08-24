"""
Telugu phrasing templates for Krishi-Agent's Orchestrator (Phase 3.2).

Template library is the PRIMARY phrasing path — locked decision after
qwen2.5:7b-instruct failed fluent Telugu generation in testing (garbled
output, stray English words). Templates are hand-written and reviewed
by the team, keyed on (resolution, confidence) from conflict_resolver.py's
output. An LLM fallback (Indic-tuned model, e.g. Sarvam AI) remains a
possible future upgrade — not yet implemented, tracked as a follow-up.
"""

TEMPLATES = {
    ("delay_treatment", "Medium"): "వర్షం రాబోతోంది, కాబట్టి వర్షం ఆగిన తర్వాత మందు కొట్టండి — చికిత్స మానకండి.",
    ("harvest_now", "High"): "మీ పంటను ఇప్పుడే కోయండి — ఇది పంటను కాపాడటానికి మరియు మంచి ధర పొందటానికి సరైన సమయం.",
    ("harvest_now", "Medium"): "మీ పంటను ఇప్పుడు కోయడం మంచిదని అనిపిస్తుంది, వర్షం మరియు ధరను దృష్టిలో ఉంచుకుని.",
    ("treat_first", "Medium"): "ధర గురించి ఆలోచించే ముందు, మీ పంటకు చికిత్స చేయడం మంచిది.",
    ("protect_crop", "Medium"): "ముందుగా మీ పంటను కాపాడుకోవడంపై దృష్టి పెట్టండి — నీటిపారుదల లేదా కోత; ధర తర్వాత చూడవచ్చు.",
    ("no_conflict", "High"): "ప్రస్తుతానికి అంతా బాగానే ఉంది — మీ పంటను గమనిస్తూ ఉండండి.",
    ("single_agent_action", "High"): "కరువు ప్రమాదం ఉన్నందున నీటిపారుదల చేయండి — ప్రస్తుతానికి వేరే సమస్య ఏమీ లేదు.",
    ("treat_and_irrigate", "Medium"): "వెంటనే పంటకు చికిత్స చేయండి, వీలైతే నీరు కూడా పెట్టండి.",
    ("treat_with_caution", "Low"): "ఇప్పుడు చికిత్స చేయండి, కానీ వాతావరణ సూచన ఖచ్చితంగా లేదని గమనించండి.",
    ("treat_now_synergy", "High"): "ఇప్పుడు చికిత్స చేయడానికి మంచి సమయం — వాతావరణం కూడా అనుకూలంగా ఉంది.",
    ("unresolved_conflict", "Low"): "ఈ విషయంలో మాకు స్పష్టమైన సమాధానం లేదు — దయచేసి మీ స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
}

# Fallback sentence if a (resolution, confidence) pair has no template
# (e.g. a resolution occurs at a confidence level not yet covered).
FALLBACK_PHRASE = "ఈ విషయంలో మాకు స్పష్టమైన సమాధానం లేదు — దయచేసి మీ స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."


def phrase_resolution(resolution, confidence):
    """
    Turn a conflict_resolver.py resolution dict into a final Telugu
    sentence via the template library. Falls back to a safe uncertain
    phrase if no template matches.
    """
    key = (resolution, confidence)
    template = TEMPLATES.get(key, FALLBACK_PHRASE)

    return template
