"""Shared prompt fragments for all reviewer agents."""

JSON_SPEC = (
    'Return ONLY valid JSON, no markdown fences, in exactly this shape:\n'
    '{"score": <number 0-10>, "summary": "<max 40 words>", '
    '"strengths": ["<max 20 words>", "... up to 3"], '
    '"issues": [{"severity": "high|medium|low", "issue": "<max 30 words>", '
    '"evidence": "<short quote or location from the document>", '
    '"fix": "<max 30 words>", "source": "<guideline tag like S1, or empty>"}]}\n'
    'Max 5 issues, most important first. Score guide: 9-10 defence-ready, 7-8 minor fixes, '
    '5-6 major gaps, below 5 fundamentally weak. Judge only what is in the excerpt; if something '
    'is absent say "not found in excerpt" rather than inventing content. '
    'Cite guideline tags only when you actually relied on them.'
)


def guideline_block(hits):
    if not hits:
        return "OFFICIAL GUIDELINES: none retrieved. Use general good academic practice."
    lines = [f"[{h['tag']}] ({h['doc']} - {h['section']}): {h['text']}" for h in hits]
    return "OFFICIAL GUIDELINES (retrieved):\n" + "\n".join(lines)
