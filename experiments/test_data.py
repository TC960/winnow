"""
Test fixture for the MMR-vs-baseline compression eval.

NARRATION: a deliberately rambly, spoken-style first-person story. It has TWO
kinds of content on purpose:

  * SEMANTIC REDUNDANCY: one idea ("the compass pointed to the lighthouse, not
    north") is restated ~4 different ways, and the grandfather relationship is
    re-described several times. This is the meaning-level repetition MMR should
    collapse and that LLMLingua-2's token pruning does NOT target.

  * SINGLE-MENTION UNIQUE FACTS: small facts (year 1947, the dog Biscuit, the
    Wren river, the island Sable, etc.) each appear exactly once. These are the
    facts a compressor can silently destroy. The hypothesis is that by removing
    the redundant restatements first, MMR frees token budget so these survive.

INTENT: drives the question-aware BGE reranker (same for both arms).

QA: comprehension questions with gold answer aliases for deterministic scoring.
Each question targets a fact; `single_mention=True` marks facts stated only once
(the ones most at risk under aggressive compression).
"""

INTENT = "Understand the key people, places, objects, and events in the narrator's story about the compass."

INSTRUCTION = (
    "You are answering reading-comprehension questions using ONLY the provided context. "
    "Answer in as few words as possible. If the answer is not present in the context, "
    "respond with exactly: UNKNOWN."
)

NARRATION = """\
So, okay, let me tell you about this thing, this compass, because honestly it's the one object I'd grab if the house were on fire. My name's Eli, by the way, Eli, and I grew up in this little town called Ashford, kind of a nothing town, you know, but it's home. Ashford. That's where all of this happened.

The compass itself is old, it's this heavy old brass compass, kind of tarnished, and the thing is, it was my grandfather's. My grandfather gave it to me. His name was Tomas, and he was, well, he was a sailor, he worked on a cargo ship for most of his life, hauling freight up and down the coast. So the compass came from him, from my grandpa Tomas, the sailor. He got it back in 1947, that's the year, 1947, when he first shipped out.

Now here's the strange part, the part I keep coming back to. The needle on this compass doesn't point north. It never pointed north. Basically the whole idea is that the needle ignores north completely and instead it always points toward the old lighthouse. Like, no matter where you stand, the needle swings around and aims straight at the lighthouse. That's the thing about it, right, it's a compass that doesn't care about north, it only cares about that lighthouse. The concept, if you want to put it simply, is that this is a compass that points to a lighthouse instead of to the pole.

And the lighthouse, I should say, is called Marrow Point. Marrow Point lighthouse, out on the headland. We used to walk there, me and my dog. Oh, the dog, I had this scruffy little terrier named Biscuit, best dog ever, and Biscuit would come with us on these walks along the water.

We'd follow the Wren, that's the river that runs through Ashford, the Wren river, down to where it meets the sea. Every autumn the town does this lantern festival, the autumn lantern festival, where everybody floats paper lanterns down the Wren at night, and it's genuinely beautiful.

My grandfather, Tomas, he made me promise him something before he passed. He made me promise that one day I'd take the compass and actually sail out to this island, an island called Sable, that he never got to reach himself. Sable Island. That was the dream, the unfinished thing, the place the compass was supposedly always trying to point us toward, in his mind anyway.

So that's the story, more or less. A brass compass from a sailor named Tomas, a needle that chases a lighthouse instead of the north pole, and a promise about an island I still haven't sailed to. One day, though. One day I'll go.
"""

# Each question: gold answer aliases (any match counts), and whether the fact is
# stated only once in the narration (single_mention -> most at risk).
QA = [
    {"q": "What is the narrator's name?", "answers": ["Eli"], "single_mention": False},
    {"q": "What town did the narrator grow up in?", "answers": ["Ashford"], "single_mention": False},
    {"q": "What object did the grandfather give the narrator?", "answers": ["a brass compass", "brass compass", "compass"], "single_mention": False},
    {"q": "What was the grandfather's name?", "answers": ["Tomas"], "single_mention": False},
    {"q": "In what year did the grandfather get the compass?", "answers": ["1947"], "single_mention": True},
    {"q": "What was the grandfather's job?", "answers": ["sailor", "cargo ship worker", "worked on a cargo ship"], "single_mention": False},
    {"q": "What is the name of the narrator's dog?", "answers": ["Biscuit"], "single_mention": True},
    {"q": "What is the name of the lighthouse?", "answers": ["Marrow Point", "Marrow Point lighthouse"], "single_mention": True},
    {"q": "What river runs through the town?", "answers": ["the Wren", "Wren", "Wren river"], "single_mention": True},
    {"q": "What festival does the town hold in autumn?", "answers": ["lantern festival", "autumn lantern festival", "the lantern festival"], "single_mention": True},
    {"q": "What island did the narrator promise to sail to?", "answers": ["Sable", "Sable Island"], "single_mention": True},
    {"q": "What does the compass needle point toward instead of north?", "answers": ["the lighthouse", "lighthouse", "Marrow Point", "Marrow Point lighthouse"], "single_mention": False},
]
