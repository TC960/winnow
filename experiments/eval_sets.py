"""
Two-task eval set for (LLMLingua-2 + reranker) with/without sentence dedup.

TASK A — "ramble": a sparse, low-density spoken monologue. Few unique facts
spread thin among filler and repeated meaning. This is the realistic voice-
transcript case. Hypothesis: semantic dedup safely removes the redundant filler
restatements without touching the sparse facts (the regime where dedup helps).

TASK B — "multidoc": several medium-density documents (8 passages, incl. 2
distractors). A broad query. The reranker has a real job here — drop the
distractors and surface the content docs — and dedup can collapse cross/intra-
doc redundancy. This is the multi-document regime, not a dense single doc.

Each task carries comprehension QA with gold aliases for deterministic scoring.
`single_mention` marks facts stated once (most at risk under compression).
"""

# =====================================================================
# TASK A — sparse spoken ramble (single doc)
# =====================================================================
RAMBLE_INTENT = "Understand the key facts about the speaker's planned camping trip."
RAMBLE_INSTRUCTION = (
    "Answer the reading-comprehension question using ONLY the provided context. "
    "Answer in as few words as possible. If the answer is not present, respond "
    "with exactly: UNKNOWN."
)
RAMBLE_TEXT = """\
Okay so, I dunno, I've just been really needing to get away lately, like just get out of the city for a bit, you know? It's been a lot. So I'm thinking, this weekend, we're gonna do a little camping trip, finally. Like actually do it this time. I just need to unplug, get away from it all, honestly, just disconnect for a couple days.

So the plan, I guess, is we're heading up to Pinecrest Lake. Pinecrest. It's, um, it's a bit of a drive, like I think it's around three hours or so to get up there, which isn't bad, it's fine, we'll just put on some music. And it's me and Dana going, Dana's coming, which'll be fun, we haven't hung out in forever.

We're gonna leave Saturday morning, like early-ish, get a head start on the day. And honestly I just want to relax, like sit by the water, maybe do some fishing. I think there's trout up there, so we'll try to catch some trout, we'll see, I'm not exactly an expert or anything but it's chill.

Last time, oh my god, last time we tried to do this we forgot the tent, like we literally drove all the way out and didn't have a tent, it was a disaster. So this time I'm triple-checking the tent, the tent is the number one thing, cannot forget the tent again. Anyway. It'll be nice. Just, like, nature, no phones, no emails, get away from all the noise for a bit.

We'll probably head back Sunday night, so it's just the one night really, short and sweet. But yeah. I'm looking forward to it, it'll be good to just recharge, get out of my own head for a minute.
"""
RAMBLE_QA = [
    {"q": "Where is the speaker going camping?", "answers": ["Pinecrest Lake", "Pinecrest"], "single_mention": False},
    {"q": "How long is the drive?", "answers": ["three hours", "3 hours", "around three hours"], "single_mention": True},
    {"q": "Who is going with the speaker?", "answers": ["Dana"], "single_mention": False},
    {"q": "What day are they leaving?", "answers": ["Saturday", "Saturday morning"], "single_mention": True},
    {"q": "What kind of fish do they hope to catch?", "answers": ["trout"], "single_mention": False},
    {"q": "What did they forget last time?", "answers": ["the tent", "tent"], "single_mention": False},
    {"q": "When are they heading back?", "answers": ["Sunday night", "Sunday"], "single_mention": True},
]

# =====================================================================
# TASK B — multi-doc, medium density (8 passages; docs 6 & 7 are distractors)
# =====================================================================
MULTIDOC_INTENT = (
    "Answer questions about the Nimbus weather-data company: its product, "
    "pricing, architecture, operations, incidents, and policies."
)
MULTIDOC_INSTRUCTION = RAMBLE_INSTRUCTION

MULTIDOC_DOCS = [
    # 0 — Product overview
    "Nimbus Product Overview. Nimbus is a weather-data API company that provides "
    "developers with programmatic access to forecasts, historical climate records, "
    "and severe-weather alerts. The company was founded in 2021 and is headquartered "
    "in Denver, Colorado. Nimbus was started by a small team of meteorologists and "
    "engineers who felt existing weather APIs were unreliable and poorly documented. "
    "The current CEO is Lena Ortiz, who previously led data platforms at a logistics "
    "firm. Nimbus serves customers ranging from solo hobbyist developers to large "
    "agriculture and aviation companies that depend on accurate, low-latency weather "
    "data. The company's stated mission is to make trustworthy weather data accessible "
    "to any developer with a few lines of code.",

    # 1 — Pricing & plans
    "Nimbus Pricing and Plans. Nimbus offers three tiers. The Free tier includes up to "
    "1,000 API calls per day at no cost and is intended for prototyping and small "
    "personal projects. The Pro plan costs $49 per month and raises the limit "
    "substantially while adding priority support and access to historical data going "
    "back ten years. The Enterprise plan is custom-priced and adds a dedicated account "
    "manager, custom SLAs, and on-premise deployment options. Customers who exceed "
    "their plan's included calls are billed for overage at a rate of $0.002 per "
    "additional call. Annual billing is available and gives a discount equivalent to "
    "two months free compared to paying monthly.",

    # 2 — Architecture
    "Nimbus Architecture Notes. The Nimbus platform ingests raw observations from three "
    "independent satellite providers and blends them with ground-station readings to "
    "produce its forecasts. The data pipeline refreshes every 15 minutes, so customers "
    "always receive recent conditions. Nimbus commits to a 99.95% uptime SLA for paying "
    "customers. The primary data store is a PostgreSQL cluster, and a Redis layer caches "
    "the most frequently requested forecasts to keep response times low. The forecasting "
    "models themselves run on a fleet of GPU workers that retrain nightly on the latest "
    "observations. All traffic is served through a global content-delivery network to "
    "minimize latency for international customers.",

    # 3 — Incident report
    "Nimbus Incident Report: March 3, 2024. On March 3, 2024, Nimbus experienced a "
    "partial outage that lasted 47 minutes, during which roughly a third of forecast "
    "requests returned errors. The root cause was traced to a bug in the Redis failover "
    "logic: when the primary cache node was restarted for maintenance, the replica did "
    "not promote correctly, and requests that missed the cache overwhelmed the database. "
    "The on-call engineer Raj Patel identified the issue and rolled back the faulty "
    "configuration to restore service. As a follow-up, the team added automated failover "
    "tests and improved alerting so the same failure mode is caught before reaching "
    "production again.",

    # 4 — API reference
    "Nimbus API Reference. The Nimbus REST API exposes three primary endpoints: "
    "/forecast returns upcoming conditions, /historical returns past observations, and "
    "/alerts returns active severe-weather warnings for a location. Every request must "
    "be authenticated by passing your API key in the X-Nimbus-Key request header; "
    "requests without a valid key are rejected. The API enforces a rate limit of 60 "
    "requests per minute per key, and clients that exceed it receive a 429 response. "
    "Responses are returned as JSON, and all timestamps are in UTC. Pagination is "
    "supported on the /historical endpoint via a cursor parameter.",

    # 5 — HR / onboarding
    "Nimbus Onboarding Handbook. Welcome to Nimbus. The company is remote-first, with "
    "team members spread across several time zones. The daily engineering standup is "
    "held at 10:00 AM Mountain Time over video. New employees receive 20 days of paid "
    "time off per year, which begins accruing on the first day. The team coordinates "
    "primarily through Slack for discussion and Linear for tracking work. New engineers "
    "are paired with an onboarding buddy for their first month and are expected to ship "
    "a small change to production within their first two weeks as part of getting "
    "familiar with the deployment process.",

    # 6 — DISTRACTOR: general weather blog (no Nimbus facts)
    "Understanding Seasonal Weather Patterns. Weather varies enormously across seasons, "
    "and understanding the broad drivers can help anyone plan ahead. In many temperate "
    "regions, spring brings increased rainfall as warming air holds more moisture, while "
    "late summer often sees thunderstorms driven by surface heating. Coastal areas tend "
    "to have milder swings than inland regions because large bodies of water moderate "
    "temperature. Meteorologists rely on a mix of satellite imagery, radar, and "
    "ground-based sensors to track these systems. For the casual observer, simply "
    "watching cloud types and wind shifts can offer surprisingly good short-term hints "
    "about incoming changes in the weather.",

    # 7 — DISTRACTOR: competitor landscape (no Nimbus gold facts)
    "The Weather-API Competitive Landscape. The market for weather data has grown "
    "crowded over the past decade. Several large incumbents offer broad global coverage "
    "bundled with mapping products, while a number of smaller startups compete on "
    "developer experience, niche data sets, or pricing. Open-data government sources "
    "remain popular for budget-conscious projects, though they often lack support and "
    "guaranteed uptime. Buyers typically evaluate providers on coverage, update "
    "frequency, documentation quality, and cost predictability. Analysts expect "
    "continued consolidation as demand for hyper-local forecasting grows among "
    "logistics, agriculture, and energy customers.",
]
# Indices of distractor docs the reranker SHOULD drop.
MULTIDOC_DISTRACTOR_INDICES = [6, 7]
# Keep the top-6 of 8 so the reranker must shed exactly the 2 distractors.
MULTIDOC_TOP_K = 6

MULTIDOC_QA = [
    {"q": "Who is the CEO of Nimbus?", "answers": ["Lena Ortiz", "Lena"], "single_mention": True},
    {"q": "In what year was Nimbus founded?", "answers": ["2021"], "single_mention": True},
    {"q": "How much does the Pro plan cost per month?", "answers": ["$49", "49 dollars", "$49 per month"], "single_mention": True},
    {"q": "What is the overage cost per extra API call?", "answers": ["$0.002", "0.002 dollars", "$0.002 per call"], "single_mention": True},
    {"q": "How often does the data pipeline refresh?", "answers": ["every 15 minutes", "15 minutes"], "single_mention": True},
    {"q": "What uptime SLA does Nimbus promise?", "answers": ["99.95%", "99.95 percent"], "single_mention": True},
    {"q": "What database does Nimbus use as its primary store?", "answers": ["PostgreSQL", "Postgres"], "single_mention": True},
    {"q": "How long did the March 3 2024 outage last?", "answers": ["47 minutes"], "single_mention": True},
    {"q": "What HTTP header authenticates API requests?", "answers": ["X-Nimbus-Key"], "single_mention": True},
    {"q": "What is the API rate limit?", "answers": ["60 requests per minute", "60 req/min", "60 requests/minute"], "single_mention": True},
    {"q": "How many paid time off days do new employees get?", "answers": ["20 days", "20"], "single_mention": True},
]

TASKS = {
    "ramble": {
        "kind_input": "text",
        "intent": RAMBLE_INTENT,
        "instruction": RAMBLE_INSTRUCTION,
        "text": RAMBLE_TEXT,
        "top_k": None,           # single doc: keep all, just compress
        "qa": RAMBLE_QA,
        "distractor_indices": [],
    },
    "multidoc": {
        "kind_input": "documents",
        "intent": MULTIDOC_INTENT,
        "instruction": MULTIDOC_INSTRUCTION,
        "documents": MULTIDOC_DOCS,
        "top_k": MULTIDOC_TOP_K,
        "qa": MULTIDOC_QA,
        "distractor_indices": MULTIDOC_DISTRACTOR_INDICES,
    },
}
