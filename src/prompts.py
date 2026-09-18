# src/prompts.py

PROMPTS = [
    # --- Factual / explanatory ---
    {"id": "fact_01", "category": "factual", "text": "Explain how vaccines train the immune system."},
    {"id": "fact_02", "category": "factual", "text": "Explain why the sky appears blue during the day."},
    {"id": "fact_03", "category": "factual", "text": "Explain how a blockchain keeps a shared ledger secure."},
    {"id": "fact_04", "category": "factual", "text": "Explain what causes ocean tides."},
    {"id": "fact_05", "category": "factual", "text": "Explain how neural networks learn from data."},
    {"id": "fact_06", "category": "factual", "text": "Explain why leaves change color in autumn."},

    # --- Creative / narrative ---
    {"id": "creative_01", "category": "creative", "text": "Write a short story about a lighthouse keeper who finds a message in a bottle."},
    {"id": "creative_02", "category": "creative", "text": "Write a short story about a chess game between two old rivals."},
    {"id": "creative_03", "category": "creative", "text": "Write a short story about a city where it never stops raining."},
    {"id": "creative_04", "category": "creative", "text": "Write a short story about a robot learning to paint."},
    {"id": "creative_05", "category": "creative", "text": "Write a short story about the last librarian on Earth."},
    {"id": "creative_06", "category": "creative", "text": "Write a short story about two strangers stuck in an elevator."},

    # --- Persuasive / opinion ---
    {"id": "persuade_01", "category": "persuasive", "text": "Argue for or against a four-day work week."},
    {"id": "persuade_02", "category": "persuasive", "text": "Argue for or against banning single-use plastics."},
    {"id": "persuade_03", "category": "persuasive", "text": "Argue for or against mandatory national service."},
    {"id": "persuade_04", "category": "persuasive", "text": "Argue for or against remote learning replacing in-person classes."},
    {"id": "persuade_05", "category": "persuasive", "text": "Argue for or against self-driving cars becoming the default mode of transport."},
    {"id": "persuade_06", "category": "persuasive", "text": "Argue for or against social media age restrictions."},

    # --- Instructional ---
    {"id": "instruct_01", "category": "instructional", "text": "Explain step by step how to change a car tire."},
    {"id": "instruct_02", "category": "instructional", "text": "Explain step by step how to make a basic omelette."},
    {"id": "instruct_03", "category": "instructional", "text": "Explain step by step how to set up a budget spreadsheet."},
    {"id": "instruct_04", "category": "instructional", "text": "Explain step by step how to plant a vegetable garden."},
    {"id": "instruct_05", "category": "instructional", "text": "Explain step by step how to back up files to an external drive."},
    {"id": "instruct_06", "category": "instructional", "text": "Explain step by step how to prepare for a job interview."},

    # --- Dialogue / conversational ---
    {"id": "dialogue_01", "category": "dialogue", "text": "Write a conversation between a customer and a barista about ordering a complicated drink."},
    {"id": "dialogue_02", "category": "dialogue", "text": "Write a conversation between a student and a teacher discussing a late assignment."},
    {"id": "dialogue_03", "category": "dialogue", "text": "Write a conversation between two friends planning a road trip."},
    {"id": "dialogue_04", "category": "dialogue", "text": "Write a conversation between a doctor and a nervous patient before a checkup."},
    {"id": "dialogue_05", "category": "dialogue", "text": "Write a conversation between a job candidate and an interviewer."},
    {"id": "dialogue_06", "category": "dialogue", "text": "Write a conversation between two siblings arguing about chores."},

    # --- Summarization (same source text given to every model) ---
    {"id": "summary_01", "category": "summary", "text": (
        "Summarize the following paragraph in 3-4 sentences:\n\n"
        "The Industrial Revolution, which began in Britain in the late 18th century, "
        "transformed economies that had been based on agriculture and handicrafts into "
        "economies based on large-scale industry, mechanized manufacturing, and the factory "
        "system. New machines, new power sources, and new ways of organizing work made "
        "existing industries more productive and efficient. It also caused rapid urbanization "
        "as workers moved from rural areas to cities to work in factories, radically reshaping "
        "social structures and living conditions."
    )},
    {"id": "summary_02", "category": "summary", "text": (
        "Summarize the following paragraph in 3-4 sentences:\n\n"
        "Coral reefs are among the most biodiverse ecosystems on the planet, hosting roughly "
        "a quarter of all marine species despite covering less than one percent of the ocean "
        "floor. They form when colonies of tiny animals called coral polyps secrete calcium "
        "carbonate skeletons over generations, building up complex structures over centuries. "
        "Rising ocean temperatures cause coral bleaching, where corals expel the symbiotic algae "
        "that give them color and nutrients, often leading to widespread die-offs if the stress "
        "persists."
    )},
    {"id": "summary_03", "category": "summary", "text": (
        "Summarize the following paragraph in 3-4 sentences:\n\n"
        "Microfinance refers to small loans, savings accounts, and other financial services "
        "offered to individuals who lack access to traditional banking, often in developing "
        "economies. The model gained global attention after the Grameen Bank in Bangladesh "
        "demonstrated that even very small loans, when paired with community accountability "
        "structures, could help low-income entrepreneurs start businesses and improve their "
        "livelihoods. Critics argue that some microfinance programs have led borrowers into "
        "cycles of debt, particularly when interest rates are high or loans are poorly regulated."
    )},
]

assert len({p["id"] for p in PROMPTS}) == len(PROMPTS), "duplicate prompt ids"

if __name__ == "__main__":
    print(f"Total prompts: {len(PROMPTS)}")
    from collections import Counter
    print(Counter(p["category"] for p in PROMPTS))