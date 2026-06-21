"""
Prompts transcribed from the AttentionRAG paper (arXiv:2503.10720), Appendix B.

These are kept verbatim (modulo whitespace) so the implementation stays faithful
to the paper. Three prompts are used:

  * HINT_PREFIX_PROMPT  (B.1) -- turns a question into an incomplete-answer
    template whose blank is the "focal token". Run on the hint-prefix model
    (GPT-4o Mini in the paper; configurable here).
  * FIXED_HINT_PREFIX   (B.2) -- the static fallback prefix used for tasks with
    no natural answer template (e.g. summarization) or when (B.1) returns 'None'.
  * ANCHOR_PROMPT       (B.3) -- wraps a single context chunk + question + hint
    and is fed to the compression model, which predicts ONE token (the anchor /
    focal token a_j) and emits 'none' when the chunk is irrelevant.
"""

# --- B.1: answer-hint-prefix generation (few-shot, format-only) -------------
HINT_PREFIX_PROMPT = """You are a formatting assistant. Given a question, your task is to generate a corresponding answering format. The format should maintain the same structure as the question but transform it into an incomplete answer template. If it is impossible to generate a format, return 'None'.

The format is like an complete answer, but truncated before the key word, and the key word is not included in the format.

For instance, if the question is 'Where is Daniel?', the format should be 'Daniel is in the', as the next word is the key word.

Note: For yes/no questions, such as 'Is Tom here?', return 'None' because these questions are typically answered with 'yes' or 'no' and do not have a natural continuation that leads to a single keyword.

Examples:
1. Question: Where is Daniel? -> Format: Daniel is in the
2. Question: What time is it? -> Format: It is
3. Question: Who is responsible for this? -> Format: The person responsible for this is
4. Question: Which film was released more recently, Dance With A Stranger or Miley Naa Miley Hum? -> Format: The film released more recently was
5. Question: Is Tom here? -> Format: None

In generation, you should only return the format, not any other text.
Now, here's a new question:
Question: {question}
Format:"""

# --- B.2: fixed fallback hint prefix ----------------------------------------
FIXED_HINT_PREFIX = (
    "Please output the most relevant keyword or phrase that is relevant to the "
    "answer of the question."
)

# --- B.3: per-chunk anchor-token prompt -------------------------------------
# The compression model continues after "Answer: {prefix}". Its first generated
# token is the anchor token a_j. If the chunk is irrelevant it emits 'none'.
ANCHOR_PROMPT = """You will be given a long context begin with 'Context:', a question begin with 'Question:', and a hint begin with 'Hint:'. Please answer the question.
Context: {context}
Hint: You should answer begin with {prefix}, if there is no useful information in the context for the question in the context and you really don't know the answer, just answer {prefix} none.
Question: {question}
Answer: {prefix}"""

# The ANCHOR_PROMPT is split into the three segments below so the compression
# backend can tokenize them independently and know the EXACT token positions of
# the context chunk (needed to read off attention over context tokens). The
# context chunk is inserted between PRE and POST.
ANCHOR_PRE = (
    "You will be given a long context begin with 'Context:', a question begin "
    "with 'Question:', and a hint begin with 'Hint:'. Please answer the "
    "question.\nContext: "
)


def build_anchor_segments(context: str, question: str, prefix: str):
    """Return (pre, ctx, post) text segments for the anchor prompt.

    `prefix` is the (possibly empty) answer hint prefix. When empty (yes/no
    questions), nothing is appended after "Answer:" and the hint line is phrased
    without a prefix to begin with.
    """
    prefix = prefix.strip()
    if prefix:
        hint = (
            f"You should answer begin with {prefix}, if there is no useful "
            "information in the context for the question in the context and you "
            f"really don't know the answer, just answer {prefix} none."
        )
        tail = f"Answer: {prefix}"
    else:
        hint = (
            "If there is no useful information in the context for the question "
            "in the context and you really don't know the answer, just answer "
            "none."
        )
        tail = "Answer:"
    post = f"\nHint: {hint}\nQuestion: {question}\n{tail}"
    return ANCHOR_PRE, context, post


# --- Final-answer prompt (LongBench-style) ----------------------------------
# Not from the paper's appendix; a standard RAG answer prompt for the
# generation model. Kept simple and greedy.
ANSWER_PROMPT = (
    "Answer the question based only on the given passages. Keep the answer short.\n\n"
    "Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
)
