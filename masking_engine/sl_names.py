"""
sl_names.py — Shared Sri Lankan Name Reference Data
R26-CS-012: Context-Aware Masking + Instruction Engine

Single source of truth for the surname pool, imported by BOTH
engine/detector.py (NER matching) and data/generate_dataset.py
(synthetic data generation), so the two can never drift apart
the way they previously did (detector recognized only 10 of the
75 surnames the generator actually used).
"""

SL_LAST_NAMES = [
    "Nadarajah", "Nagahawatta", "Nagahawatte", "Nagel", "Nagendra", "Nalliah",
    "Napier", "Nanayakkara", "Jayakody", "Jayaratne", "Jayasekera",
    "Ilangakoon", "Illesinghe", "Isbooldeniya", "Hamilton", "Gooneratne",
    "Goonesekera", "Goonetilleke", "Goonewardena", "Goonewardene",
    "Edirisinghe", "Ekanayake", "Fernand", "Fernandez", "Fernando",
    "Alvis", "Alwines", "Alwis", "Amarasuriya", "Amarasekera",
    "Amaratunga", "Ambrose", "Ameresekera", "Ameresekere", "Anandappa",
    "Anderson", "Andrews", "Dias", "Bandranayaka", "Baptist", "Barber",
    "Dharmaratne", "Dissanayake", "Dassenaike", "Jayasinghe", "Obeyesekera",
    "Obeysekere", "Paranagama", "Paranavitarne", "Ramanathan", "Ramalingam",
    "Ranasinghe", "Ranatunga", "Ranawake", "Ranaweerasinghe", "Salgado",
    "Salvador", "Samarakkody", "Samarakoon", "Samaranayake", "Samarasekera",
    "Samarasinghe", "Perera", "Silva", "Jayawardena", "Wickramasinghe",
    "Gunawardena", "Rajapaksa", "Bandara", "Senanayake", "Rajapaksha",
    "Wickremasinghe", "Mendis", "Karunaratne", "Weerasinghe",
]
