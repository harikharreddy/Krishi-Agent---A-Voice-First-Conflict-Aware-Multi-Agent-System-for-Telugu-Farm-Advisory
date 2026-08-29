"""
Phase 5.2 -- Telugu numeral-to-words preprocessing for TTS.
Phase 6 -- improved natural Telugu number naming (hundreds/thousands),
replacing raw num2words output which used literal/unnatural phrasing
(e.g. "ఒకటి వేయిల యాభై ఒకటి" instead of "వెయ్యి యాభై ఒకటి" for 1051).
Also strips trailing whitespace num2words sometimes returns (e.g.
num2words(50, lang="te") == "యాభై " with a trailing space), which was
producing double spaces in normalized output.

Converts embedded digits in agent/template answer text into Telugu
words before handing text to the TTS model, since ai4bharat/indic-parler-tts
was found (Phase 5.2 testing) to mispronounce multi-digit numbers and
percentages when left as raw digits.

Handles two patterns seen in real agent output:
  - "2450 రూ."  -> "<words> రూ."
  - "29%"       -> "<words> శాతం"
"""

import re
from num2words import num2words


def _telugu_number_words(n: int) -> str:
    if n == 0:
        return "సున్నా"

    parts = []
    thousands, rem = divmod(n, 1000)
    if thousands:
        if thousands == 1:
            parts.append("వెయ్యి")
        else:
            parts.append(f"{num2words(thousands, lang='te').strip()} వేలు")

    hundreds, rem2 = divmod(rem, 100)
    if hundreds:
        if hundreds == 1:
            parts.append("వంద")
        else:
            parts.append(f"{num2words(hundreds, lang='te').strip()} వందలు")

    if rem2:
        parts.append(num2words(rem2, lang="te").strip())

    return " ".join(parts)


def _number_to_telugu_words(match: re.Match) -> str:
    number_str = match.group(1)
    has_percent = match.group(2) == "%"
    number = int(number_str)
    words = _telugu_number_words(number)
    return f"{words} శాతం" if has_percent else words


def normalize_numerals_te(text: str) -> str:
    """
    Replace digit sequences (optionally followed by %) with natural
    Telugu words. Leaves all other text untouched.
    """
    pattern = re.compile(r"(\d+)(%?)")
    return pattern.sub(_number_to_telugu_words, text)


if __name__ == "__main__":
    test_cases = [
        "ఈరోజు Tomato ధర 2450 రూ., ఈ నెలలో సాధారణ ధర (1900 రూ.) కంటే 29% ఎక్కువ. ఇప్పుడు అమ్మడం మంచిది.",
        "ఈరోజు Potato ధర 820 రూ., ఈ నెలలో సాధారణ ధర (980 రూ.) కంటే 16% తక్కువ. వీలైతే ఆగడం మంచిది.",
        "రాబోయే 3 గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత 65%).",
        "రాబోయే 24 గంటల్లో వర్షం పడే అవకాశం ఉంది (వర్షం సంభావ్యత 45%).",
        "ఈరోజు Potato ధర 1051 రూ.",
        "ఈరోజు Potato ధర 9999 రూ.",
        "ఈరోజు Potato ధర 500 రూ.",
    ]
    for t in test_cases:
        print("BEFORE:", t)
        print("AFTER: ", repr(normalize_numerals_te(t)))
        print()
