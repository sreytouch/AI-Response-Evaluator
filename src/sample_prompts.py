"""Evaluation tasks with enough context to try immediately."""

SAMPLE_PROMPTS = [
    {"title": "Gravity for a child", "prompt": "Explain gravity to a 7-year-old in no more than 3 sentences.", "constraints": "Use everyday words. Do not exceed 3 sentences."},
    {"title": "Photosynthesis", "prompt": "Explain photosynthesis to a middle-school student using simple language and one example.", "constraints": "Include exactly one example."},
    {"title": "Polite rewrite", "prompt": 'Rewrite “Send me the report today” to sound more polite without changing the meaning.', "constraints": "Preserve today's deadline."},
    {"title": "Weather vs. climate", "prompt": "Explain the difference between weather and climate in under 100 words.", "constraints": "Use fewer than 100 words."},
    {"title": "Machine learning", "prompt": "Give a beginner-friendly explanation of machine learning without technical jargon.", "constraints": "Assume no programming background."},
    {"title": "Why the sky is blue", "prompt": "Explain why the sky appears blue to a 12-year-old.", "constraints": "Use an age-appropriate explanation."},
    {"title": "Extension request", "prompt": "Write a professional email asking a professor for a two-day extension, under 120 words.", "constraints": "Use fewer than 120 words and ask for exactly two extra days."},
    {"title": "Exercise benefits", "prompt": "Give three benefits of exercise without making medical claims.", "constraints": "Exactly three benefits. Do not promise to prevent or treat illness."},
    {"title": "Two-sentence summary", "prompt": "Summarize the following paragraph in exactly two sentences:\n\nA neighborhood library started a weekend seed exchange. Residents can take seeds, leave seeds from their gardens, and share growing tips. Volunteers label every packet with planting instructions. After the first month, the library added workshops about container gardening for residents without outdoor space.", "constraints": "Use exactly two sentences and only facts in the paragraph."},
    {"title": "Answer from a passage", "prompt": "Answer the question using only the supplied passage.\n\nPassage: The community center opens at 9 a.m. on weekdays and 10 a.m. on Saturdays. It is closed on Sundays. The art room can be booked for two hours at a time.\n\nQuestion: When does the center open on Saturdays, and how long can someone book the art room?", "constraints": "Use only the passage. If information is missing, say so."},
]
