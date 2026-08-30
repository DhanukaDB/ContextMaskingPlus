"""
generate_negative_corpus.py — Open-Domain Negative Examples for Layer 2 Training
R26-CS-012: Context-Aware Masking + Instruction Engine

WHY THIS FILE EXISTS (see docs/Comprehensive Technical Documentation.md
Section 8.3 for the full write-up): the ML safety-net classifier
(engine/ml_anomaly.py) was found to score ordinary, unrelated sentences
as 95%+ "sensitive." Root cause: its negative-class (label=0) training
examples came ONLY from data/generate_dataset.py's EDGE_POOL "should-not-
mask" templates — a handful of narrow taxonomy-specific shapes (serial
numbers, tracking references, names-alone, IPs-alone). The model learned
to recognize THAT SPECIFIC TEMPLATE FAMILY, not genuine non-sensitivity,
and retraining on 33% more of the SAME template family didn't help.

This generator produces a much larger, topically and structurally diverse
pool of genuinely open-domain prompts — greetings, scheduling, general
tech/how-to questions, workplace/HR chat, and banking-ADJACENT-but-safe
process questions (branch hours, transfer timelines) that carry zero PII
or secrets. All labeled contains_sensitive=0.

WHY THIS IS A SEPARATE FILE, NOT MERGED INTO synthetic_dataset.json:
  synthetic_dataset.json is the Layer-1 evaluation dataset with a panel-
  mandated size range (5,000-5,500 prompts, see generate_dataset.py).
  Adding ~2,000 more rows here would blow past that requirement for no
  Layer-1 benefit (Layer 1's regex/NER has no plausible reason to
  false-fire on "what time does the branch open on Saturdays"). This
  corpus is consumed only by research/retrain_classifier.py, which
  merges it with the taxonomy dataset's own negative examples purely for
  Layer 2's training set.
"""

import json
import os
import random

random.seed(20260830)  # fixed seed — this corpus should be reproducible, unlike the Layer-1 dataset

# ─────────────────────────────────────────────
# VOCABULARY POOLS
# ─────────────────────────────────────────────

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["9am", "9:30am", "10am", "10:30am", "11am", "noon", "1pm", "2pm",
         "2:30pm", "3pm", "3:45pm", "4pm", "5pm", "5:30pm", "end of day"]
CITIES = ["Colombo", "Kandy", "Galle", "Jaffna", "Kurunegala", "Negombo",
          "Matara", "Anuradhapura", "Batticaloa", "Ratnapura",
          "the head office", "the branch", "the regional office"]
TOPICS = ["the quarterly report", "the onboarding checklist", "the release notes",
          "the sprint retro", "the marketing deck", "the vendor contract",
          "the office relocation", "the annual audit", "the team offsite",
          "the budget forecast", "the training schedule", "the client proposal",
          "the hiring plan", "the product roadmap", "the incident postmortem",
          "the compliance checklist", "the vendor onboarding", "the style guide"]
LANGUAGES = ["Python", "JavaScript", "SQL", "Java", "TypeScript", "Go", "C#", "Rust"]
UI_ELEMENTS = ["the submit button", "the dropdown menu", "the navigation bar",
               "the search box", "the login page", "the settings panel",
               "the checkout page", "the filter panel", "the sidebar",
               "the notifications icon", "the profile menu", "the export button"]
FOODS = ["lunch", "dinner", "coffee", "a snack", "breakfast", "tea"]
WEATHER = ["sunny", "rainy", "cloudy", "humid", "windy", "cool", "warm", "overcast"]
SPORTS = ["cricket", "football", "badminton", "rugby", "swimming", "volleyball", "tennis"]
NAMES_GENERIC = ["Alex", "Sam", "Jordan", "Taylor", "Priya", "Kasun", "Nadia",
                  "Morgan", "Riley", "Casey", "the new intern", "the team lead",
                  "the project manager", "the new hire"]
DEPARTMENTS = ["Engineering", "Marketing", "Finance", "Operations", "Customer Support",
               "HR", "Legal", "Compliance", "Product"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge"]
DEVICES = ["mobile", "tablet", "desktop", "an iPad"]

# ─────────────────────────────────────────────
# TEMPLATE CATEGORIES — deliberately varied register, length, and structure
# ─────────────────────────────────────────────

TEMPLATES = [
    # Greetings / small talk
    lambda: f"Good morning! How was your weekend?",
    lambda: f"Hey, do you have a minute to chat about {random.choice(TOPICS)}?",
    lambda: f"Thanks for the quick reply earlier, really appreciate it.",
    lambda: f"Happy Friday everyone, hope you all have a great weekend.",
    lambda: f"It was {random.choice(WEATHER)} all day today, perfect for a walk.",
    lambda: f"Did you catch the {random.choice(SPORTS)} match last night?",
    lambda: f"Let's grab {random.choice(FOODS)} sometime this week and catch up.",
    lambda: f"Welcome to the team, {random.choice(NAMES_GENERIC)}! Let us know if you need anything.",
    lambda: f"{random.choice(NAMES_GENERIC)} mentioned {random.choice(TOPICS)} might slip a few days.",
    lambda: f"Congrats on the {random.choice(SPORTS)} tournament win last weekend!",

    # Scheduling / calendar
    lambda: f"Can we move our {random.choice(DAYS)} meeting to {random.choice(TIMES)}?",
    lambda: f"Is {random.choice(TIMES)} on {random.choice(DAYS)} still good for you?",
    lambda: f"I'll be out of office on {random.choice(DAYS)}, can we reschedule?",
    lambda: f"Please confirm the room booking for {random.choice(DAYS)} afternoon.",
    lambda: f"What time does the {random.choice(CITIES)} office open on {random.choice(DAYS)}s?",
    lambda: f"Can we push the {random.choice(TOPICS)} review to {random.choice(TIMES)} on {random.choice(DAYS)}?",
    lambda: f"{random.choice(NAMES_GENERIC)} from {random.choice(DEPARTMENTS)} wants to join the {random.choice(DAYS)} call.",

    # General workplace / HR (no PII)
    lambda: f"How many annual leave days do we get this year?",
    lambda: f"Where can I find the updated expense policy?",
    lambda: f"Who should I contact about setting up a new laptop?",
    lambda: f"Is there a dress code for the client visit next week?",
    lambda: f"Can someone point me to the parking permit application form?",
    lambda: f"What's the process for booking annual leave?",
    lambda: f"Which floor is the {random.choice(DEPARTMENTS)} team sitting on now?",
    lambda: f"Does {random.choice(DEPARTMENTS)} have any open roles this quarter?",

    # General product / how-to (non-security)
    lambda: f"How do I export {random.choice(TOPICS)} as a PDF?",
    lambda: f"Can you walk me through how to duplicate a spreadsheet tab?",
    lambda: f"What's the keyboard shortcut to undo in this editor?",
    lambda: f"How do I change the default currency shown on the dashboard?",
    lambda: f"Is there a dark mode option for this app?",
    lambda: f"How do I pin {random.choice(UI_ELEMENTS)} so it stays visible while scrolling?",

    # General tech support, NOT involving secrets
    lambda: f"{random.choice(UI_ELEMENTS)} isn't responding on {random.choice(DEVICES)}, any ideas why?",
    lambda: f"The page takes a long time to load, is that a known issue?",
    lambda: f"Why does {random.choice(UI_ELEMENTS)} look different on {random.choice(BROWSERS)} than {random.choice(BROWSERS)}?",
    lambda: f"The build keeps failing on the CI server, can you take a look?",
    lambda: f"What's the difference between REST and GraphQL for this kind of API?",
    lambda: f"Can you explain what this function in {random.choice(LANGUAGES)} does at a high level?",
    lambda: f"How do I write a for loop that skips every other item in {random.choice(LANGUAGES)}?",
    lambda: f"The app crashes when I rotate the screen on {random.choice(DEVICES)}, is that reproducible on your end?",
    lambda: f"Can we bump the timeout on the health check endpoint?",
    lambda: f"Is the staging environment currently down for maintenance?",
    lambda: f"Does {random.choice(LANGUAGES)} or {random.choice(LANGUAGES)} make more sense for this microservice?",
    lambda: f"{random.choice(UI_ELEMENTS)} looks misaligned on {random.choice(DEVICES)}, can someone check the CSS?",

    # Banking-adjacent but carrying zero PII/secrets — the hardest, most
    # useful negatives, since they're topically close to what this product
    # actually handles without containing anything sensitive at all.
    lambda: f"What are the working hours of the {random.choice(CITIES)} branch on public holidays?",
    lambda: f"How long does a standard wire transfer usually take to clear?",
    lambda: f"Do we charge a fee for early loan settlement?",
    lambda: f"What documents does a new customer generally need to open an account?",
    lambda: f"Is the mobile banking app down for anyone else right now?",
    lambda: f"What's the current base interest rate on a fixed deposit?",
    lambda: f"Can you summarize the new KYC policy in a few bullet points?",
    lambda: f"How often does the core banking system get its maintenance window?",
    lambda: f"What's the difference between a savings account and a current account?",
    lambda: f"Are ATMs at the {random.choice(CITIES)} branch open 24 hours?",
    lambda: f"Does the {random.choice(CITIES)} branch have a queue-ticket system?",
    lambda: f"What's the daily ATM withdrawal limit for a standard debit card?",
    lambda: f"How do I set up a standing order through online banking, in general terms?",
    lambda: f"Is there a minimum balance requirement for a student savings account?",

    # Short imperative / question fragments (structural diversity)
    lambda: f"Please review the attached slide deck before Friday.",
    lambda: f"Any update on the {random.choice(TOPICS)}?",
    lambda: f"Let me know if this looks good to you.",
    lambda: f"Can you double check this before it goes out?",
    lambda: f"Sounds good, talk soon.",
    lambda: f"Thanks, that resolves my question.",
    lambda: f"Following up on my earlier message — any news?",
    lambda: f"Sure, I can take care of that this afternoon.",
    lambda: f"Noted, I'll loop {random.choice(NAMES_GENERIC)} in as well.",
    lambda: f"Can you resend {random.choice(TOPICS)}, I think the link expired?",

    # Longer, more descriptive sentences (structural diversity)
    lambda: (f"We were discussing {random.choice(TOPICS)} in the standup this morning and a few people "
              f"raised questions about the timeline, so it might be worth revisiting the plan before the next check-in."),
    lambda: (f"The weather in {random.choice(CITIES)} has been {random.choice(WEATHER)} all week, "
              f"which has made the commute a bit unpredictable for a lot of the team."),
    lambda: (f"I spent most of the afternoon going through {random.choice(TOPICS)} and think it's in "
              f"good shape overall, though a couple of sections could use another pass."),
    lambda: (f"{random.choice(NAMES_GENERIC)} from {random.choice(DEPARTMENTS)} asked whether we could "
              f"walk through {random.choice(TOPICS)} together sometime {random.choice(DAYS)} or {random.choice(DAYS)}."),
    lambda: (f"The {random.choice(CITIES)} office has been renovating the lobby, so visitors have been "
              f"asked to use the side entrance until the {random.choice(TOPICS)} wraps up."),
]


def generate(count: int) -> list:
    prompts = set()
    attempts = 0
    while len(prompts) < count and attempts < count * 20:
        attempts += 1
        text = random.choice(TEMPLATES)()
        prompts.add(text)
    return sorted(prompts)


if __name__ == "__main__":
    target = 2000
    prompts = generate(target)

    records = [
        {"id": f"NEG_{i:04d}", "prompt": p, "entities": [], "source": "open_domain_negative_corpus"}
        for i, p in enumerate(prompts, start=1)
    ]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negative_corpus.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} open-domain negative examples -> {out_path}")
