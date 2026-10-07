import re


def normalize_word(word):
    return re.sub(r"^[^\w]+|[^\w]+$", "", word.strip().lower())


def split_prompt_words(prompt):
    return [word for raw in prompt.strip().split() if (word := normalize_word(raw))]


def find_word_span(tokens, word, start_from=0):
    word = word.lower()
    for index in range(start_from, len(tokens)):
        token = tokens[index]
        if token.lower() == "\u2581" + word or token.lstrip("\u2581").lower() == word:
            return [index], index + 1
    stripped = [token.lstrip("\u2581").lower() for token in tokens]
    for index in range(start_from, len(tokens)):
        text = ""
        span = []
        for end in range(index, len(tokens)):
            text += stripped[end]
            span.append(end)
            if text == word:
                return span, end + 1
            if len(text) > len(word):
                break
    return [], start_from


def map_entity_tokens(prompt, annotation, tokenizer, max_length):
    positions = [int(value) for value in annotation.get("entity", [])]
    words = split_prompt_words(prompt)
    if not positions or any(value < 1 or value > len(words) for value in positions):
        raise ValueError(f"Invalid entity word positions: {positions}")
    targets = [words[value - 1] for value in positions]
    encoded = tokenizer(
        [prompt], padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0].tolist())
    indices, misses, cursor = [], [], 0
    for word in targets:
        span, cursor = find_word_span(tokens, word, cursor)
        if not span:
            misses.append(word)
        indices.extend(span)
    if not indices:
        raise ValueError(f"No core entity words could be mapped to tokenizer tokens: {targets}")
    return {
        "positions": positions,
        "words": targets,
        "token_indices": list(dict.fromkeys(indices)),
        "tokens": tokens,
        "misses": misses,
    }
