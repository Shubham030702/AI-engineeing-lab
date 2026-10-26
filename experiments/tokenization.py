import tiktoken


# A public tokenizer encoding supported by tiktoken.
# This is a tokenizer experiment, not an API call.
encoding = tiktoken.get_encoding("o200k_base")
second_encoding = tiktoken.get_encoding("cl100k_base")

examples = {
    "simple English": "Hello, how are you?",
    "technical terms": "Tokenization, embeddings, transformer architecture",
    "code": "def add(a, b):\n    return a + b",
    "repeated text": "AI " * 20,
    "multilingual": "Hello नमस्ते こんにちは",
}

for label, text in examples.items():
    token_ids = encoding.encode(text)
    decoded_text = encoding.decode(token_ids)
    sec_token_ids = second_encoding.encode(text)
    sec_decoded_text = second_encoding.decode(sec_token_ids)
    print(f"\n--- {label} ---")
    print("Text:", repr(text))
    print("Character count:", len(text))
    print("Token count:", len(token_ids))
    print("Token count (2nd):", len(sec_token_ids))
    print("\nToken IDs:", token_ids)
    print("\nsec_token_ids:", sec_token_ids)
    print("\nDecoded text:", decoded_text)
    print("Decoded correctly:", decoded_text == text)
    print("\nsec_decoded_text:", sec_decoded_text)
    print("sec_decoded correctly:", sec_decoded_text == text)



def calculate_context_budget(
    context_limit: int,
    system_tokens: int,
    history_tokens: int,
    retrieved_tokens: int,
    question_tokens: int,
    reserved_output_tokens: int,
) -> dict:
    input_tokens = (
        system_tokens
        + history_tokens
        + retrieved_tokens
        + question_tokens
    )

    total_allocated = input_tokens + reserved_output_tokens
    remaining = context_limit - total_allocated

    return {
        "input_tokens": input_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "total_allocated": total_allocated,
        "remaining_tokens": remaining,
        "fits": remaining >= 0,
    }


# budget = calculate_context_budget(
#     context_limit=8192,
#     system_tokens=500,
#     history_tokens=1200,
#     retrieved_tokens=2500,
#     question_tokens=200,
#     reserved_output_tokens=1000,
# )

# print("\n--- Context budget ---")
# for key, value in budget.items():
#     print(f"{key}: {value}")