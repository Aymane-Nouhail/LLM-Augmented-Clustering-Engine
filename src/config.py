import os
from dotenv import load_dotenv

load_dotenv()

# --- API Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print(
        "Warning: OPENAI_API_KEY not set. LLM generation methods (keyphrase, pairwise, correction) will be unavailable."
    )

# --- Embedding Backend ---
# "sentence_transformers": use all-mpnet-base-v2 locally (recommended for replication)
# "openai": use text-embedding-ada-002 via OpenAI API
EMBEDDING_BACKEND = "sentence_transformers"
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"

EMBEDDING_MODEL_NAME = (
    "text-embedding-ada-002"  # used only when EMBEDDING_BACKEND="openai"
)
GENERATION_MODEL_NAME = "gpt-4.1-nano"
MOCKING_MODE = False

# --- Clustering Parameters ---
PC_NUM_PAIRS_TO_QUERY = 20000
PC_CONSTRAINT_SELECTION_STRATEGY = "explore_consolidate"  # 'random', 'similarity', or 'explore_consolidate'

CORRECTION_K_LOW_CONFIDENCE = 100
CORRECTION_NUM_CANDIDATE_CLUSTERS = 3

CLUSTERLLM_N_TRIPLETS = 1024
CLUSTERLLM_N_PAIRWISE = 400

# --- Data Loading & Results ---
# All experiment outputs (embeddings cache, keyphrase cache, query logs, metrics) live here.
# Structure: results/{dataset}/embeddings.pkl, keyphrases_cache.pkl, pairwise_queries.csv, ...
RESULTS_ROOT = "results/"
DATA_CACHE_PATH = RESULTS_ROOT  # embeddings are cached under results/{dataset}/


# =============================================================================
# Per-dataset prompt templates
# =============================================================================
# Each dataset has 3 prompts: keyphrase (KP), pairwise (PC), correction.
# Tailored to the domain: Bank77 = banking intents, CLINC = broad conversational
# AI, Tweet = short-text topic classification (Yin & Wang 2016, 89 categories).
# =============================================================================

# ---------------------------------------------------------------------------
# BANK77 — 77 banking/financial intent categories
# ---------------------------------------------------------------------------
BANK77_KP_PROMPT_TEMPLATE = """
You are an expert in banking and financial services.

Task: Extract 5 to 7 keyphrases that capture the user's banking intent and the specific financial action or topic they are asking about. Focus on the type of transaction, account feature, card operation, or financial service involved.

- Use concise noun phrases or verb phrases (e.g., "card activation", "international transfer fee", "direct debit setup").
- Avoid generic words like "bank", "account", "money" unless they are part of a specific compound phrase.
- Avoid full sentences.

Output format: JSON list of strings.
"""

BANK77_PC_PROMPT_TEMPLATE = """
Determine if the following two banking queries express the same user intent (i.e., the user wants to accomplish the same banking task or get the same type of information).

Focus on the specific financial action or service, not surface wording.

Examples:

Query 1: how do I activate my card
Query 2: my new card isn't working, how do I turn it on
Answer ONLY with YES or NO.
YES

Query 1: what are the fees for sending money abroad
Query 2: how do I set up a direct debit
Answer ONLY with YES or NO.
NO

Query 1: I want to freeze my card
Query 2: can I lock my card temporarily
Answer ONLY with YES or NO.
YES

Query 1: how do I check my balance
Query 2: what's my transaction history
Answer ONLY with YES or NO.
NO

Now, determine the following:

Query 1: {text1}
Query 2: {text2}
Answer ONLY with YES or NO.
"""

BANK77_CORRECTION_PROMPT_TEMPLATE = """
You are an expert in banking and financial services intent classification.

Task: Decide whether the following banking query belongs to the same intent category as the representative query from the cluster.

Examples:

User Query: how do I activate my new card
Representative Query: activate debit card
Same banking intent?
YES

User Query: what is the exchange rate for euros
Representative Query: cancel my card
Same banking intent?
NO

User Query: I need to report a lost card
Representative Query: my card has been stolen
Same banking intent?
YES

Now evaluate:

User Query: {document_text}
Representative Query: {rep_doc_text}
Same banking intent?

Respond ONLY with YES or NO.
"""

# ---------------------------------------------------------------------------
# CLINC — 150 conversational AI intent categories across many domains
# ---------------------------------------------------------------------------
CLINC_KP_PROMPT_TEMPLATE = """
You are an expert in understanding user intent in conversational AI assistants.

Task: Extract 5 to 7 keyphrases that capture the user's intent and goal in the following utterance. The assistant handles a wide range of tasks: weather, alarms, reminders, travel, shopping, calendar, small talk, and more.

- Focus on what the user wants to do or know.
- Include the action type and the domain (e.g., "set alarm", "weather forecast", "flight booking", "restaurant reservation").
- Use concise noun phrases or verb phrases.
- Avoid full sentences.

Output format: JSON list of strings.
"""

CLINC_PC_PROMPT_TEMPLATE = """
Determine if the following two user utterances express the same intent (i.e., the user is trying to accomplish the same goal or get the same type of information from a virtual assistant).

Focus only on the underlying intent, not surface wording or specific details.

Examples:

Utterance 1: what is my balance
Utterance 2: how much money do I have in my account
Answer ONLY with YES or NO.
YES

Utterance 1: set an alarm for 7am
Utterance 2: wake me up at seven in the morning
Answer ONLY with YES or NO.
YES

Utterance 1: cancel my flight
Utterance 2: book me a flight to Paris
Answer ONLY with YES or NO.
NO

Utterance 1: what's the weather like today
Utterance 2: will it rain tomorrow in London
Answer ONLY with YES or NO.
NO

Utterance 1: tell me a joke
Utterance 2: say something funny
Answer ONLY with YES or NO.
YES

Now, determine the following:

Utterance 1: {text1}
Utterance 2: {text2}
Answer ONLY with YES or NO.
"""

CLINC_CORRECTION_PROMPT_TEMPLATE = """
You are an expert in conversational AI intent classification.

Task: Decide whether the following user utterance belongs to the same intent category as the representative utterance. The system covers many domains (weather, alarms, travel, shopping, banking, small talk, etc.).

Examples:

User Utterance: remind me to call John at 3pm
Representative Utterance: set a reminder for my meeting tomorrow
Same intent?
YES

User Utterance: what time is it in Tokyo
Representative Utterance: what's the weather in New York
Same intent?
NO

User Utterance: I lost my card
Representative Utterance: my card was stolen
Same intent?
YES

Now evaluate:

User Utterance: {document_text}
Representative Utterance: {rep_doc_text}
Same intent?

Respond ONLY with YES or NO.
"""

# ---------------------------------------------------------------------------
# TWEET (Yin & Wang 2016) — 89 short-text topic categories
# These are topic/theme labels, not conversational intents.
# ---------------------------------------------------------------------------
TWEET_KP_PROMPT_TEMPLATE = """
You are an expert in social media content analysis.

Task: Extract 5 to 7 keyphrases that capture the main topic, theme, and key concepts of the following tweet. Tweets are short and informal; focus on identifying the subject matter and domain (e.g., sports event, political news, health advice, entertainment, technology product).

- Use concise noun phrases (e.g., "NBA playoffs", "vaccination campaign", "movie release", "stock market crash").
- Include both specific entities and broader thematic concepts.
- Avoid filler words and generic terms.
- Avoid full sentences.

Output format: JSON list of strings.
"""

TWEET_PC_PROMPT_TEMPLATE = """
Determine if the following two tweets are about the same topic or theme (i.e., they belong to the same subject-matter category, even if they discuss different specific events or details).

Focus on the broader topic domain, not surface wording or specific named entities.

Examples:

Tweet 1: Just watched an amazing touchdown at the Super Bowl!
Tweet 2: The Lakers won last night in overtime, incredible game
Answer ONLY with YES or NO.
YES

Tweet 1: New iPhone announced with better camera
Tweet 2: NASA's new Mars rover just landed successfully
Answer ONLY with YES or NO.
NO

Tweet 1: Scientists warn about rising ocean temperatures
Tweet 2: Record heat wave hits Europe this summer
Answer ONLY with YES or NO.
YES

Tweet 1: Best restaurants to try in NYC this weekend
Tweet 2: Oscar nominations announced for 2024
Answer ONLY with YES or NO.
NO

Now, determine the following:

Tweet 1: {text1}
Tweet 2: {text2}
Answer ONLY with YES or NO.
"""

TWEET_CORRECTION_PROMPT_TEMPLATE = """
You are an expert in short-text topic classification for social media.

Task: Decide whether the following tweet belongs to the same topic category as the representative tweet from the cluster. Focus on the broad subject-matter theme, not specific details.

Examples:

Tweet: The championship game goes into double overtime
Representative Tweet: World Cup finals draw record viewership
Same topic?
YES

Tweet: New drug approved for treating diabetes
Representative Tweet: Apple unveils new MacBook lineup
Same topic?
NO

Tweet: Mayor announces new public transit funding
Representative Tweet: Senate passes infrastructure bill
Same topic?
YES

Now evaluate:

Tweet: {document_text}
Representative Tweet: {rep_doc_text}
Same topic?

Respond ONLY with YES or NO.
"""

# ---------------------------------------------------------------------------
# BANK77 — normalization + paraphrase
# ---------------------------------------------------------------------------
BANK77_NORM_PROMPT_TEMPLATE = """Rewrite the following banking query in clean, canonical form.
Use standard financial terminology. Remove filler words, slang, and informal language.
Be concise (5-15 words). Output ONLY the rewritten text, nothing else.

Query: {text}"""

BANK77_PARAPHRASE_PROMPT_TEMPLATE = """Generate 3 paraphrases of the following banking query.
Each paraphrase should express the same intent using different wording.
Output as a JSON list of strings.

Query: {text}"""

# ---------------------------------------------------------------------------
# CLINC — normalization + paraphrase
# ---------------------------------------------------------------------------
CLINC_NORM_PROMPT_TEMPLATE = """Rewrite the following conversational assistant utterance in clean, canonical form.
Remove filler words and informal language while preserving the core intent.
Be concise (5-15 words). Output ONLY the rewritten text, nothing else.

Utterance: {text}"""

CLINC_PARAPHRASE_PROMPT_TEMPLATE = """Generate 3 paraphrases of the following utterance.
Each paraphrase should express the same intent using different wording.
Output as a JSON list of strings.

Utterance: {text}"""

# ---------------------------------------------------------------------------
# TWEET — normalization + paraphrase
# ---------------------------------------------------------------------------
TWEET_NORM_PROMPT_TEMPLATE = """Rewrite the following tweet in clean, canonical form.
Remove hashtags, emojis, slang, and informal language. Preserve the core topic using formal vocabulary.
Be concise (5-15 words). Output ONLY the rewritten text, nothing else.

Tweet: {text}"""

TWEET_PARAPHRASE_PROMPT_TEMPLATE = """Generate 3 paraphrases of the following tweet.
Each paraphrase should describe the same topic using different wording (no hashtags or emojis).
Output as a JSON list of strings.

Tweet: {text}"""

# ---------------------------------------------------------------------------
# ClusterLLM — triplet prompts (Stage 1 perspective queries)
# ---------------------------------------------------------------------------
BANK77_CLUSTERLLM_TRIPLET_PROMPT = """You are an expert in banking and financial services.

Anchor Query: {anchor}

Option A: {option_a}
Option B: {option_b}

Which option (A or B) expresses the same banking intent as the Anchor Query?
Respond with ONLY the letter A or B."""

CLINC_CLUSTERLLM_TRIPLET_PROMPT = """You are an expert in conversational AI intent classification.

Anchor Utterance: {anchor}

Option A: {option_a}
Option B: {option_b}

Which option (A or B) expresses the same intent as the Anchor Utterance?
Respond with ONLY the letter A or B."""

TWEET_CLUSTERLLM_TRIPLET_PROMPT = """You are an expert in social media content analysis.

Anchor Tweet: {anchor}

Option A: {option_a}
Option B: {option_b}

Which option (A or B) is about the same topic as the Anchor Tweet?
Respond with ONLY the letter A or B."""

# ---------------------------------------------------------------------------
# Routing: maps dataset name → its prompt dict
# ---------------------------------------------------------------------------
DATASET_PROMPTS = {
    "bank77": {
        "kp": BANK77_KP_PROMPT_TEMPLATE,
        "pc": BANK77_PC_PROMPT_TEMPLATE,
        "correction": BANK77_CORRECTION_PROMPT_TEMPLATE,
        "normalization": BANK77_NORM_PROMPT_TEMPLATE,
        "paraphrase": BANK77_PARAPHRASE_PROMPT_TEMPLATE,
        "triplet": BANK77_CLUSTERLLM_TRIPLET_PROMPT,
    },
    "clinc": {
        "kp": CLINC_KP_PROMPT_TEMPLATE,
        "pc": CLINC_PC_PROMPT_TEMPLATE,
        "correction": CLINC_CORRECTION_PROMPT_TEMPLATE,
        "normalization": CLINC_NORM_PROMPT_TEMPLATE,
        "paraphrase": CLINC_PARAPHRASE_PROMPT_TEMPLATE,
        "triplet": CLINC_CLUSTERLLM_TRIPLET_PROMPT,
    },
    "tweet": {
        "kp": TWEET_KP_PROMPT_TEMPLATE,
        "pc": TWEET_PC_PROMPT_TEMPLATE,
        "correction": TWEET_CORRECTION_PROMPT_TEMPLATE,
        "normalization": TWEET_NORM_PROMPT_TEMPLATE,
        "paraphrase": TWEET_PARAPHRASE_PROMPT_TEMPLATE,
        "triplet": TWEET_CLUSTERLLM_TRIPLET_PROMPT,
    },
}
