"""
Dataset Generator — R26-CS-012
Context-Aware Masking + Instruction Engine

Generates 5000–5500 synthetic prompts (random count each run)
covering taxonomy categories 1–7 including adversarial and edge/ambiguity cases.

Enhanced with:
  - Full Sri Lankan first name pool (160+ names)
  - Full Sri Lankan surname pool (70+ surnames)
  - All licensed banks in Sri Lanka (local + foreign branches)
  - All major Sri Lankan cities by province
  - Expanded prompt template variety per category
"""

import random
import csv
import json
import base64
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sl_names import SL_LAST_NAMES

random.seed(random.randint(1, 99999))  # non-fixed seed → different count each run

# ─────────────────────────────────────────────
# SRI LANKAN REFERENCE DATA
# ─────────────────────────────────────────────

SL_FIRST_NAMES = [
    "Mohamed", "Kasun", "Indika", "Gayan", "Nuwan", "Amila", "Chaminda",
    "Janaka", "Roshan", "Ruwan", "Sampath", "Dinesh", "Asanka", "Sameera",
    "Saman", "Pradeep", "Ajith", "Tharindu", "Sanjeewa", "Chandana",
    "Mahesh", "Lahiru", "Aruna", "Manjula", "Buddhika", "Isuru", "Sarath",
    "Fathima", "Prasanna", "Rasika", "Mohammed", "Priyantha", "Fernando",
    "Nishantha", "Harsha", "Nalin", "Thilina", "Upul", "Thushara", "Gamini",
    "Manoj", "Anura", "Dhanushka", "Thilini", "Chamara", "Ranjith", "Chanaka",
    "Abdul", "Samantha", "Chinthaka", "Rohan", "Sanjaya", "Lalith", "Chathura",
    "Wasantha", "Jagath", "Sunil", "Suranga", "Gihan", "Tharanga", "Prasad",
    "Chamila", "Suresh", "Jayantha", "Nimal", "Anuradha", "Ravi", "Hasitha",
    "Lasantha", "Supun", "Chandima", "Charith", "Kamal", "Kanchana", "Thusitha",
    "Prabath", "Duminda", "Nalaka", "Tharaka", "Dhammika", "Damith", "Ananda",
    "Kapila", "Shehan", "Dinusha", "Gayani", "Rajitha", "Darshana", "Udaya",
    "Nadeesha", "Asela", "Sajith", "Anusha", "Ranga", "Udara", "Upali",
    "Sumith", "Pubudu", "Eranga", "Nihal", "Mahinda", "Ishara", "Susantha",
    "Shantha", "Sanath", "Sumudu", "Sandun", "Sujeewa", "Kanishka", "Danushka",
    "Amal", "Lakmal", "Thanuja", "Ashan", "Chathurika", "Viraj", "Nadeeka",
    "Dimuthu", "Kosala", "Ravindra", "Athula", "Niroshan", "Asoka", "Shanika",
    "Anushka", "Palitha", "Shanaka", "Hemantha", "Akila", "Chathuranga",
    "Shashika", "Bandula", "Uditha", "Jude", "Sisira", "Namal", "Iresha",
    "Waruna", "Ishan", "Lakshman", "Kushan", "Sudath", "Sujith", "Nilantha",
    "Rukshan", "Harshani", "Nishan", "Nirosha", "Gayathri", "Hiran", "Dilan",
    "Sahan", "Anil", "Inoka", "Thilak", "Ramesh", "Yohan", "Charitha",
    "Rohana", "Kumudu", "Mohan", "Ranil", "Dilshan", "Shan", "Vajira",
    "Anton", "Muditha", "Buddika", "Malind", "Priya", "Sanduni",
]

SL_BANKS = [
    # State-owned
    "Bank of Ceylon", "People's Bank",
    # Licensed commercial banks
    "Commercial Bank of Ceylon", "Hatton National Bank", "Sampath Bank",
    "Seylan Bank", "Nations Trust Bank", "DFCC Bank",
    "National Development Bank", "Pan Asia Banking Corporation",
    "Union Bank of Colombo", "Amana Bank", "Cargills Bank",
    # Foreign branches
    "HSBC", "Standard Chartered Bank", "Citibank", "Deutsche Bank",
    "Bank of China", "State Bank of India", "Indian Bank",
    "Indian Overseas Bank", "MCB Bank", "Habib Bank", "Public Bank Berhad",
]

SL_SWIFT_CODES = [
    "BCEYLKLX", "PEOHLKLX", "CCEYLKLX", "HBLILKLX", "BSAMLKLX",
    "SEYBLKLX", "NTBKLKLX", "DFCCLKLX", "HSBCLKLX", "SCBLLKLX", "CITILKLX",
]

SL_CITIES = [
    # Western Province
    "Colombo", "Sri Jayawardenepura Kotte", "Dehiwala-Mount Lavinia",
    "Moratuwa", "Negombo", "Gampaha", "Kalutara", "Maharagama",
    "Kaduwela", "Panadura", "Avissawella", "Beruwala",
    # Central Province
    "Kandy", "Matale", "Nuwara Eliya", "Gampola", "Hatton", "Dambulla",
    # Southern Province
    "Galle", "Matara", "Hambantota", "Tangalle", "Ambalangoda",
    "Weligama", "Hikkaduwa",
    # Northern Province
    "Jaffna", "Vavuniya", "Mannar", "Point Pedro", "Chavakachcheri",
    # Eastern Province
    "Trincomalee", "Batticaloa", "Kalmunai", "Ampara",
    # Other key cities
    "Kurunegala", "Anuradhapura", "Ratnapura", "Badulla",
    "Puttalam", "Polonnaruwa",
]

SL_ROADS = [
    "Galle Road", "Kandy Road", "High Level Road", "Duplication Road",
    "Baseline Road", "Negombo Road", "Matara Road", "Kurunegala Road",
    "Hospital Road", "Main Street", "Station Road", "Temple Road",
    "Colombo Road", "Rajapaksha Mawatha", "Independence Avenue",
    "Bauddhaloka Mawatha", "D.S. Senanayake Mawatha", "Flower Road",
    "Union Place", "Nawala Road", "Kotte Road", "Dehiwala Road",
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "seylan.lk", "boc.lk", "hnb.lk", "sampath.lk",
    "ntb.lk", "dfcc.lk", "combank.lk", "hattonbank.lk",
]


# ─────────────────────────────────────────────
# ENTITY GENERATORS
# ─────────────────────────────────────────────

def rand_name():
    return f"{random.choice(SL_FIRST_NAMES)} {random.choice(SL_LAST_NAMES)}"

def rand_nic_old():
    # 9 digits total (2-digit year + 3-digit day-of-year + 4-digit seq) + V/X suffix
    year   = random.randint(50, 99)
    days   = random.randint(1, 366)
    seq    = random.randint(1000, 9999)
    suffix = random.choice(["V", "X"])
    return f"{year}{days:03d}{seq}{suffix}"

def rand_nic_new():
    # 12 digits total (4-digit year + 3-digit day-of-year + 5-digit seq)
    year = random.randint(1950, 2005)
    days = random.randint(1, 366)
    seq  = random.randint(10000, 99999)
    return f"{year}{days:03d}{seq}"

def rand_passport():
    letter = random.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
    number = random.randint(1000000, 9999999)
    return f"{letter}{number}"

def rand_phone_lk():
    prefixes = ["070", "071", "072", "075", "076", "077", "078"]
    return f"{random.choice(prefixes)}{random.randint(1000000, 9999999)}"

def rand_phone_intl():
    # Country codes other than 94 (LK), so it can't collide with PHONE_LK.
    country_codes = ["1", "44", "61", "65", "971", "966", "852", "91", "49", "33"]
    cc   = random.choice(country_codes)
    body = "".join(str(random.randint(0, 9)) for _ in range(random.randint(7, 10)))
    return f"+{cc}{body}"

def rand_tax_id():
    return "".join(str(random.randint(0, 9)) for _ in range(9))

def rand_driving_license():
    letter = random.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
    number = random.randint(1000000, 9999999)
    return f"{letter}{number}"

def rand_email(name=None):
    if name is None:
        name = rand_name()
    user   = name.lower().replace(" ", random.choice([".", "_", ""]))
    suffix = str(random.randint(1, 999)) if random.random() > 0.6 else ""
    return f"{user}{suffix}@{random.choice(EMAIL_DOMAINS)}"

def _luhn_check_digit(partial_digits: str) -> str:
    digits = [int(d) for d in partial_digits]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - (total % 10)) % 10)

def rand_pan():
    prefixes = ["4", "5", "37", "6011"]
    prefix   = random.choice(prefixes)
    length   = 16 if len(prefix) < 3 else 15
    body     = prefix + "".join([str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1)])
    number   = body + _luhn_check_digit(body)
    sep      = random.choice([" ", "-", ""])
    return sep.join([number[i:i+4] for i in range(0, len(number), 4)])

def rand_cvv():
    return str(random.randint(100, 999))

def rand_expiry():
    month = random.randint(1, 12)
    year  = random.randint(25, 30)
    return random.choice([f"{month:02d}/{year}", f"{month:02d}/{2000+year}"])

def rand_bank_account():
    length = random.choice([10, 12, 14, 16])
    return "".join([str(random.randint(0, 9)) for _ in range(length)])

def rand_iban():
    # detector's IBAN regex requires: 2 letters + 2 digits + 4 alnum (bank
    # code) + 7-20 DIGITS (account digits) — a fully alphanumeric BBAN
    # (the old generator) can't structurally match it.
    prefix    = random.choice(["GB", "LK", "DE", "FR", "NL", "AE"])
    check     = random.randint(10, 99)
    bank_code = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))
    acct_digits = "".join(random.choices("0123456789", k=random.randint(7, 20)))
    return f"{prefix}{check}{bank_code}{acct_digits}"

def rand_swift():
    return random.choice(SL_SWIFT_CODES)

def rand_bank():
    return random.choice(SL_BANKS)

def rand_api_key():
    # "sk-" intentionally excluded: it's what makes the detector's more
    # specific API_KEY_OPENAI pattern fire instead of API_KEY_GENERIC, so
    # templates that expect an API_KEY_GENERIC ground truth must never
    # accidentally generate one. Use rand_api_key_openai() when an
    # sk--prefixed key is actually wanted.
    chars  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    length = random.randint(28, 40)
    prefix = random.choice(["pk-", "api-", "key-"])
    return f"{prefix}{''.join(random.choices(chars, k=length))}"

def rand_api_key_openai():
    # Real OpenAI keys run ~48-56 chars after "sk-"; the detector demotes
    # anything shorter to API_KEY_GENERIC rather than over-claiming the
    # vendor (fixes v.xlsx #5) — this generator must match that threshold
    # or its own ground truth becomes unmatchable.
    chars  = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    length = random.randint(48, 56)
    return f"sk-{''.join(random.choices(chars, k=length))}"

def rand_password():
    words    = ["Secure", "Bank", "Admin", "Pass", "Login", "Access",
                "Lanka", "Finance", "Digital", "Prime", "Shield", "Vault"]
    specials = ["@", "#", "!", "$", "%"]
    return f"{random.choice(words)}{random.randint(10, 9999)}{random.choice(specials)}"

def rand_jwt():
    header  = base64.b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
    payload = base64.b64encode(
        f'{{"sub":"{rand_bank_account()}","role":"teller","exp":9999999999}}'.encode()
    ).decode().rstrip("=")
    sig = "".join(random.choices(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", k=43))
    return f"{header}.{payload}.{sig}"

def rand_db_conn():
    engines = ["postgresql", "mysql", "mongodb", "mssql", "redis"]
    users   = ["admin", "dbuser", "bankapp", "coredb", "appuser", "readonly"]
    dbs     = ["corebank_prod", "customer_db", "transactions", "audit_log",
               "loan_system", "forex_db", "kyc_data", "cbsl_reports"]
    host    = f"10.{random.randint(0,9)}.{random.randint(0,99)}.{random.randint(1,254)}"
    port    = random.choice([5432, 3306, 27017, 1433, 6379])
    return (f"{random.choice(engines)}://{random.choice(users)}"
            f":{rand_password()}@{host}:{port}/{random.choice(dbs)}")

def rand_aws_key():
    return "AKIA" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=16))

def rand_aws_secret():
    return "".join(random.choices(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/", k=40))

def rand_s3_bucket():
    names = ["bank-backups", "customer-docs", "audit-logs", "transaction-exports",
             "kyc-uploads", "statement-archive", "cbsl-reports", "loan-docs"]
    bank_slug = rand_bank().lower().replace(" ", "-").replace("'", "")[:12]
    return f"s3://sl-{bank_slug}-{random.choice(names)}"

def rand_ip_internal():
    subnet = random.choice(["10", "192.168", "172.16"])
    return f"{subnet}.{random.randint(0,9)}.{random.randint(0,99)}.{random.randint(1,254)}"

def rand_address():
    return f"{random.randint(1,350)}, {random.choice(SL_ROADS)}, {random.choice(SL_CITIES)}"

def rand_dob():
    year  = random.randint(1955, 2002)
    month = random.randint(1, 12)
    day   = random.randint(1, 28)
    return random.choice([
        f"{year}-{month:02d}-{day:02d}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d}-{month:02d}-{year}",
    ])

def rand_amount():
    return f"LKR {random.randint(500, 10_000_000):,}"

def rand_loan_ref():
    return f"LN{random.randint(100000, 999999)}"

def rand_account_type():
    return random.choice(["savings", "current", "fixed deposit", "overdraft", "loan"])

def rand_business_ref(digits=12):
    """A generic non-sensitive business reference number (transaction/
    order/invoice/tracking id) — deliberately the SAME digit-shape as
    NIC_NEW/BANK_ACCOUNT_NO/PHONE_LK so the templates below exercise the
    context-qualifier suppression logic (fixes v.xlsx #1, #10, #11), not
    just a differently-shaped number that would trivially not match."""
    return "".join(str(random.randint(0, 9)) for _ in range(digits))

def rand_jwt_secret():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return "".join(random.choices(chars, k=random.randint(24, 40)))


# ─────────────────────────────────────────────
# ADVERSARIAL GENERATORS
# ─────────────────────────────────────────────

def adv_nic_spaced():
    return " ".join(list(rand_nic_old()))

def adv_nic_dashed():
    nic = rand_nic_old()
    mid = random.randint(3, len(nic) - 2)
    return nic[:mid] + "-" + nic[mid:]

def adv_nic_mixed():
    # Space+dash mixed obfuscation. NOT a dot ("199012.345V") — the
    # normalizer's despacing/dash-joining logic has no dot-handling (by
    # design: blanket digit.digit stripping would corrupt IPs and decimal
    # currency amounts elsewhere in a prompt), so a dot-obfuscated NIC is
    # a real, currently-unsupported gap rather than something this
    # generator should claim as a solvable ground-truth case.
    nic = rand_nic_old()
    mid = len(nic) // 2
    return nic[:mid] + "-" + " ".join(list(nic[mid:]))

def adv_b64_api():
    """Base64-encoded API key. Ground truth depends on which generator
    produced the key: 'sk-' decodes into something that matches the more
    specific API_KEY_OPENAI pattern; everything else falls through to the
    keyword-gated API_KEY_GENERIC pattern."""
    key = random.choice([rand_api_key(), rand_api_key_openai()])
    encoded = base64.b64encode(key.encode()).decode()
    label = "API_KEY_OPENAI" if key.startswith("sk-") else "API_KEY_GENERIC"
    return encoded, label

def adv_b64_password():
    # Encode the full "password=<value>" unit, not just the bare value —
    # the PASSWORD pattern is defined as keyword+separator+value together
    # (`password\s*[=:]\s*\S+`), so a decoded bare value with no
    # "password=" prefix anywhere in the prompt can never match it.
    return base64.b64encode(f"password={rand_password()}".encode()).decode()

def adv_hex_password():
    return f"password={rand_password()}".encode().hex()

def adv_hex_nic():
    return rand_nic_old().encode().hex()

def adv_url_password():
    pw = rand_password()
    return "password%3D" + pw.replace("@", "%40").replace("#", "%23").replace("!", "%21")

def adv_url_email():
    em = rand_email()
    return em.replace("@", "%40").replace(".", "%2E")

def adv_mixed_case_key():
    # Uppercasing the first 2 chars always breaks the case-sensitive
    # "sk-" prefix match, so a mangled key can never land as
    # API_KEY_OPENAI — it always falls through to API_KEY_GENERIC.
    key = rand_api_key()
    return key[:2].upper() + key[2:5].lower() + "-" + key[5:]


# ─────────────────────────────────────────────
# PROMPT POOLS
# ─────────────────────────────────────────────

NORMAL_POOL = [

    # ── 1A NIC Old ────────────────────────────────────────────────
    ("1A-NIC-OLD", lambda: (f"Can you check the account for NIC {rand_nic_old()}? The customer called in today.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"KYC verification needed for NIC number {rand_nic_old()} — flagged by compliance.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Please pull the loan history for NIC {rand_nic_old()}.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Customer NIC: {rand_nic_old()} — update address in core system.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Our system shows NIC {rand_nic_old()} has two active accounts — please review.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"CBSL audit request: provide transaction records for NIC {rand_nic_old()}.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Blocked account holder with NIC {rand_nic_old()} is at the branch.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Verify the identity of NIC {rand_nic_old()} before processing the transfer.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Dormant account linked to NIC {rand_nic_old()} — request reactivation documents.", ["NIC_OLD"])),
    ("1A-NIC-OLD", lambda: (f"Fraud alert on NIC {rand_nic_old()} — escalate to AML team.", ["NIC_OLD"])),

    # ── 1A NIC New ────────────────────────────────────────────────
    ("1A-NIC-NEW", lambda: (f"Customer id is {rand_nic_new()}, please verify KYC status.", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"New NIC format detected: {rand_nic_new()} — is the format valid?", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"Identity number {rand_nic_new()} was submitted via the mobile app.", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"Check the blacklist for customer ID {rand_nic_new()}.", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"The customer's national ID is {rand_nic_new()} — onboard them to {rand_bank()}.", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"Duplicate NIC alert: {rand_nic_new()} appears in two customer profiles.", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"PEP screening flagged national ID {rand_nic_new()} — hold account.", ["NIC_NEW"])),
    ("1A-NIC-NEW", lambda: (f"Biometric mismatch for identity {rand_nic_new()} — reject e-KYC.", ["NIC_NEW"])),

    # ── 1A Passport ───────────────────────────────────────────────
    ("1A-PASSPORT", lambda: (f"Customer provided passport {rand_passport()} as ID — proceed with onboarding.", ["PASSPORT"])),
    ("1A-PASSPORT", lambda: (f"Foreign national with passport number {rand_passport()} wants to open an account at {rand_bank()}.", ["PASSPORT"])),
    ("1A-PASSPORT", lambda: (f"Verify passport {rand_passport()} against INTERPOL watch-list.", ["PASSPORT"])),
    ("1A-PASSPORT", lambda: (f"Passport {rand_passport()} expiry in 3 months — flag for re-KYC.", ["PASSPORT"])),

    # ── 1A Tax ID / Driving License ─────────────────────────────────
    ("1A-TAX-ID", lambda: (f"Please confirm the TIN {rand_tax_id()} for this corporate account.", ["TAX_ID"])),
    ("1A-TAX-ID", lambda: (f"VAT registration number {rand_tax_id()} needs to be updated on file.", ["TAX_ID"])),
    ("1A-TAX-ID", lambda: (f"Tax identification {rand_tax_id()} flagged for annual review.", ["TAX_ID"])),
    ("1A-DRIVING-LICENSE", lambda: (f"Customer's driving license {rand_driving_license()} was provided as secondary ID.", ["DRIVING_LICENSE"])),
    ("1A-DRIVING-LICENSE", lambda: (f"Verify driving license number {rand_driving_license()} against DMT records.", ["DRIVING_LICENSE"])),
    ("1A-DRIVING-LICENSE", lambda: (f"License {rand_driving_license()} submitted for identity verification.", ["DRIVING_LICENSE"])),

    # ── 1B Contact ────────────────────────────────────────────────
    ("1B-CONTACT", lambda: (f"Send the OTP to {rand_phone_lk()} and cc {rand_email()}.", ["PHONE_LK", "EMAIL"])),
    ("1B-CONTACT-INTL", lambda: (f"International client reachable at {rand_phone_intl()} — confirm callback time.", ["PHONE_INTL"])),
    ("1B-CONTACT-INTL", lambda: (f"Correspondent bank contact: {rand_phone_intl()} for SWIFT queries.", ["PHONE_INTL"])),
    ("1B-CONTACT-INTL", lambda: (f"Forward the wire confirmation call to {rand_phone_intl()}.", ["PHONE_INTL"])),
    ("1B-CONTACT", lambda: (f"Customer reachable at {rand_phone_lk()} or {rand_email()} — please follow up.", ["PHONE_LK", "EMAIL"])),
    ("1B-CONTACT", lambda: (f"Update contact: phone {rand_phone_lk()}, email {rand_email()}.", ["PHONE_LK", "EMAIL"])),
    ("1B-CONTACT", lambda: (f"Loan confirmation SMS sent to {rand_phone_lk()} and email to {rand_email()}.", ["PHONE_LK", "EMAIL"])),
    ("1B-CONTACT", lambda: (f"Dispatch the card PIN to {rand_phone_lk()}.", ["PHONE_LK"])),
    ("1B-CONTACT", lambda: (f"Reset password link sent to {rand_email()}.", ["EMAIL"])),
    ("1B-CONTACT", lambda: (f"Two-factor auth registered to {rand_phone_lk()} for this {rand_bank()} account.", ["PHONE_LK"])),
    ("1B-CONTACT", lambda: (f"Marketing opt-out request from {rand_email()} — update preferences.", ["EMAIL"])),
    ("1B-CONTACT", lambda: (f"Transaction alert failed to reach {rand_phone_lk()} — fallback to {rand_email()}.", ["PHONE_LK", "EMAIL"])),
    ("1B-CONTACT", lambda: (f"Complaint registered under {rand_email()} for account issue at {rand_bank()}.", ["EMAIL"])),

    # ── 1C Demographics ───────────────────────────────────────────
    ("1C-DEMOGRAPHICS", lambda: (f"Update the profile for {rand_name()}, DOB {rand_dob()}, address {rand_address()}.", ["FULL_NAME", "DATE_OF_BIRTH", "HOME_ADDRESS"])),
    ("1C-DEMOGRAPHICS", lambda: (f"KYC file for {rand_name()}: born {rand_dob()}, residing at {rand_address()}.", ["FULL_NAME", "DATE_OF_BIRTH", "HOME_ADDRESS"])),
    ("1C-DEMOGRAPHICS", lambda: (f"Customer {rand_name()}, address {rand_address()} — send monthly statement.", ["FULL_NAME", "HOME_ADDRESS"])),
    ("1C-DEMOGRAPHICS", lambda: (f"Deceased account holder: {rand_name()}, DOB {rand_dob()} — freeze account.", ["FULL_NAME", "DATE_OF_BIRTH"])),
    ("1C-DEMOGRAPHICS", lambda: (f"Name: {rand_name()} | Address: {rand_address()} | Phone: {rand_phone_lk()}.", ["FULL_NAME", "HOME_ADDRESS", "PHONE_LK"])),
    ("1C-DEMOGRAPHICS", lambda: (f"The beneficiary {rand_name()} lives at {rand_address()} — verify before disbursement.", ["FULL_NAME", "HOME_ADDRESS"])),
    ("1C-DEMOGRAPHICS", lambda: (f"Mortgage applicant: {rand_name()}, born {rand_dob()}, currently at {rand_address()}.", ["FULL_NAME", "DATE_OF_BIRTH", "HOME_ADDRESS"])),
    ("1C-DEMOGRAPHICS", lambda: (f"Student loan for {rand_name()}, DOB {rand_dob()} — confirm eligibility.", ["FULL_NAME", "DATE_OF_BIRTH"])),

    # ── 2A Card Data ──────────────────────────────────────────────
    ("2A-CARD", lambda: (f"Card number {rand_pan()}, CVV {rand_cvv()}, expiry {rand_expiry()} — is this Luhn valid?", ["PAN", "CVV", "CARD_EXPIRY"])),
    ("2A-CARD", lambda: (f"Disputed transaction on card {rand_pan()} — CVV on file is {rand_cvv()}.", ["PAN", "CVV"])),
    ("2A-CARD", lambda: (f"Block card {rand_pan()} immediately — customer reports theft.", ["PAN"])),
    ("2A-CARD", lambda: (f"Card {rand_pan()} expiring {rand_expiry()} — trigger renewal workflow.", ["PAN", "CARD_EXPIRY"])),
    ("2A-CARD", lambda: (f"3DS auth failed for {rand_pan()} at {rand_bank()} merchant terminal.", ["PAN"])),
    ("2A-CARD", lambda: (f"Chargeback raised for card {rand_pan()}, CVV {rand_cvv()}, used on {rand_expiry()}.", ["PAN", "CVV", "CARD_EXPIRY"])),
    ("2A-CARD", lambda: (f"International transaction blocked on {rand_pan()} — customer travelling.", ["PAN"])),
    ("2A-CARD", lambda: (f"Card replacement requested for {rand_pan()} — deliver to {rand_address()}.", ["PAN", "HOME_ADDRESS"])),
    ("2A-CARD", lambda: (f"Limit increase approved for card {rand_pan()} — update in CBS.", ["PAN"])),

    # ── 2B Bank Account ───────────────────────────────────────────
    ("2B-BANK", lambda: (f"Transfer from account {rand_bank_account()} to IBAN {rand_iban()}, SWIFT {rand_swift()}.", ["BANK_ACCOUNT_NO", "IBAN", "SWIFT_BIC"])),
    ("2B-BANK", lambda: (f"Debit {rand_amount()} from account {rand_bank_account()} at {rand_bank()}.", ["BANK_ACCOUNT_NO"])),
    ("2B-BANK", lambda: (f"Wire {rand_amount()} to IBAN {rand_iban()} via SWIFT {rand_swift()}.", ["IBAN", "SWIFT_BIC"])),
    ("2B-BANK", lambda: (f"Freeze {rand_account_type()} account {rand_bank_account()} pending AML review.", ["BANK_ACCOUNT_NO"])),
    ("2B-BANK", lambda: (f"Customer's {rand_bank()} account number is {rand_bank_account()} — link to loan {rand_loan_ref()}.", ["BANK_ACCOUNT_NO"])),
    ("2B-BANK", lambda: (f"IBAN {rand_iban()} rejected by correspondent bank {rand_swift()} — retry?", ["IBAN", "SWIFT_BIC"])),
    ("2B-BANK", lambda: (f"Reconcile account {rand_bank_account()} against {rand_bank()} statement.", ["BANK_ACCOUNT_NO"])),
    ("2B-BANK", lambda: (f"Standing order set up from {rand_bank_account()} to {rand_bank_account()} every month.", ["BANK_ACCOUNT_NO"])),
    ("2B-BANK", lambda: (f"Joint account {rand_bank_account()} at {rand_bank()} — remove secondary holder.", ["BANK_ACCOUNT_NO"])),
    ("2B-BANK", lambda: (f"Overdrawn account {rand_bank_account()} — charge penalty interest.", ["BANK_ACCOUNT_NO"])),

    # ── 2C SWIFT ─────────────────────────────────────────────────
    ("2C-SWIFT", lambda: (f"The MT103 message shows {rand_name()} transferred {rand_amount()} to account {rand_bank_account()}.", ["SWIFT_MT103", "FULL_NAME", "BANK_ACCOUNT_NO"])),
    ("2C-SWIFT", lambda: (f"MT103 received from {rand_swift()} for {rand_amount()} — post to account {rand_bank_account()}.", ["SWIFT_MT103", "SWIFT_BIC", "BANK_ACCOUNT_NO"])),
    ("2C-SWIFT", lambda: (f"MT202 bank-to-bank cover for SWIFT {rand_swift()} — settle {rand_amount()}.", ["SWIFT_MT202", "SWIFT_BIC"])),
    ("2C-SWIFT", lambda: (f"SWIFT BIC {rand_swift()} sent an MT202 to {rand_iban()} — log this for CBSL.", ["SWIFT_BIC", "IBAN", "SWIFT_MT202"])),
    ("2C-SWIFT", lambda: (f"Nostro reconciliation: MT103 ref {rand_loan_ref()} for {rand_amount()} via {rand_swift()}.", ["SWIFT_MT103", "SWIFT_BIC"])),
    ("2C-SWIFT", lambda: (f"Correspondent bank {rand_swift()} queried our MT103 — respond with account {rand_bank_account()}.", ["SWIFT_BIC", "SWIFT_MT103", "BANK_ACCOUNT_NO"])),

    # ── 3A API / JWT ──────────────────────────────────────────────
    ("3A-API", lambda: (f"I'm getting a 401 error when using api_key={rand_api_key()} against the payment gateway.", ["API_KEY_GENERIC"])),
    ("3A-API", lambda: (f"API call fails: Authorization header api_key={rand_api_key()} — wrong scope?", ["API_KEY_GENERIC"])),
    ("3A-API", lambda: (f"Token {rand_api_key()} is returning 403 on /v1/accounts endpoint.", ["API_KEY_GENERIC"])),
    ("3A-API", lambda: (f"Replace old api_key {rand_api_key()} with the new one from vault.", ["API_KEY_GENERIC"])),
    # "Authorization: Bearer <jwt>" phrasing also genuinely matches the
    # more specific JWT_IN_LOG pattern (Taxonomy 7C) — both are real.
    ("3A-API", lambda: (f"Authorization: Bearer {rand_jwt()} — why is the token rejected?", ["JWT_TOKEN", "JWT_IN_LOG"])),
    ("3A-API", lambda: (f"JWT {rand_jwt()} has expired — refresh it for session continuity.", ["JWT_TOKEN"])),
    ("3A-API", lambda: (f"Service account key: {rand_api_key()} — rotate before month end.", ["API_KEY_GENERIC"])),
    ("3A-API", lambda: (f"CBS API key {rand_api_key()} has been revoked — provision a new one.", ["API_KEY_GENERIC"])),
    ("3A-API", lambda: (f"Webhook secret is {rand_api_key()} — update in payment processor config.", ["API_KEY_GENERIC"])),

    # ── 3B Password ───────────────────────────────────────────────
    # Note: every template below uses an explicit "="/":" between the
    # keyword and value on purpose — the detector's PASSWORD pattern
    # requires one (per taxonomy 3B's own "keyword + separator + value"
    # definition), matching how a real config/log line discloses a secret.
    # A bare "password VALUE" with no separator at all is indistinguishable
    # from ordinary sentences like "password reset failed" and isn't a
    # realistic disclosure shape, so it's deliberately not a supported
    # ground-truth case here.
    ("3B-PASSWORD", lambda: (f"The login keeps failing with password: {rand_password()} — did the policy change?", ["PASSWORD"])),
    ("3B-PASSWORD", lambda: (f"Admin console password: {rand_password()} needs to be updated — CBSL audit next week.", ["PASSWORD"])),
    ("3B-PASSWORD", lambda: (f"password={rand_password()} was committed to the repo — rotate immediately.", ["PASSWORD"])),
    ("3B-PASSWORD", lambda: (f"SSH login: user=admin password={rand_password()} to {rand_ip_internal()}.", ["PASSWORD", "INTERNAL_IP"])),
    ("3B-PASSWORD", lambda: (f"DB password reset to: {rand_password()} — update all service configs.", ["PASSWORD"])),
    ("3B-PASSWORD", lambda: (f"VPN credentials: user=sysadmin pass={rand_password()} — update MFA too.", ["PASSWORD"])),

    # ── 3B Secret Assignments (env + code style) — fixes v.xlsx #3/4/6/7 ──
    ("3B-SECRET-ENV",  lambda: (f"The production configuration contains DB_PASSWORD={rand_password()}. Please identify the configuration issue.", ["PASSWORD"])),
    ("3B-SECRET-ENV",  lambda: (f"The production configuration contains JWT_SECRET={rand_jwt_secret()}. Please identify the security issue.", ["PASSWORD"])),
    ("3B-SECRET-ENV",  lambda: (f"CLIENT_SECRET={rand_jwt_secret()} was found in the deployment logs — rotate now.", ["PASSWORD"])),
    ("3B-SECRET-CODE", lambda: (f"Please review this code: const password = '{rand_password()}'; const user = 'admin';", ["PASSWORD"])),
    ("3B-SECRET-CODE", lambda: (f"Please review this code: const jwtSecret = '{rand_jwt_secret()}'; const user = 'admin';", ["PASSWORD"])),
    ("3B-SECRET-CODE", lambda: (f"Code review flagged: let apiSecret = \"{rand_jwt_secret()}\"; — should not be hardcoded.", ["PASSWORD"])),

    # ── 4B DB Connections ─────────────────────────────────────────
    ("4B-DB", lambda: (f"The app can't connect: {rand_db_conn()} — is the firewall blocking the port?", ["DB_CONNECTION_STRING", "INTERNAL_IP"])),
    ("4B-DB", lambda: (f"Connection string in config.yaml: {rand_db_conn()} — never push to public repo.", ["DB_CONNECTION_STRING", "INTERNAL_IP"])),
    ("4B-DB", lambda: (f"Backup job uses: {rand_db_conn()} — check credentials.", ["DB_CONNECTION_STRING", "INTERNAL_IP"])),
    ("4B-DB", lambda: (f"Migration script fails on: {rand_db_conn()} — is the schema updated?", ["DB_CONNECTION_STRING", "INTERNAL_IP"])),
    ("4B-DB", lambda: (f"Read replica: {rand_db_conn()} — latency is too high.", ["DB_CONNECTION_STRING", "INTERNAL_IP"])),

    # ── 4C Network ────────────────────────────────────────────────
    ("4C-NETWORK", lambda: (f"Internal server at {rand_ip_internal()} is unreachable from {rand_ip_internal()}.", ["INTERNAL_IP"])),
    ("4C-NETWORK", lambda: (f"Core banking host {rand_ip_internal()} — ping test failed.", ["INTERNAL_IP"])),
    ("4C-NETWORK", lambda: (f"Firewall rule: allow {rand_ip_internal()} to {rand_ip_internal()} on port 5432.", ["INTERNAL_IP"])),
    ("4C-NETWORK", lambda: (f"SIEM alert: brute force from {rand_ip_internal()} against {rand_ip_internal()}.", ["INTERNAL_IP"])),
    ("4C-NETWORK", lambda: (f"NAT gateway {rand_ip_internal()} dropped packets — check logs.", ["INTERNAL_IP"])),

    # ── 7A Cloud Keys ─────────────────────────────────────────────
    ("7A-CLOUD", lambda: (f"AWS_ACCESS_KEY_ID={rand_aws_key()} AWS_SECRET_ACCESS_KEY={rand_aws_secret()} — rotate now.", ["AWS_ACCESS_KEY", "AWS_SECRET_KEY"])),
    ("7A-CLOUD", lambda: (f"Found hardcoded key {rand_aws_key()} in the deployment script.", ["AWS_ACCESS_KEY"])),
    ("7A-CLOUD", lambda: (f"IAM access key {rand_aws_key()} has S3 full access — restrict permissions.", ["AWS_ACCESS_KEY"])),
    ("7A-CLOUD", lambda: (f"Leaked secret: AWS_SECRET={rand_aws_secret()} — revoke immediately.", ["AWS_SECRET_KEY"])),

    # ── 7B Storage ────────────────────────────────────────────────
    ("7B-STORAGE", lambda: (f"Backup the audit logs to {rand_s3_bucket()} before the CBSL inspection.", ["S3_BUCKET_REF"])),
    ("7B-STORAGE", lambda: (f"KYC documents uploaded to {rand_s3_bucket()} — confirm lifecycle policy.", ["S3_BUCKET_REF"])),
    ("7B-STORAGE", lambda: (f"Customer statements archived in {rand_s3_bucket()} — check access controls.", ["S3_BUCKET_REF"])),
    ("7B-STORAGE", lambda: (f"Sync {rand_s3_bucket()} to DR region — retention is 7 years.", ["S3_BUCKET_REF"])),
]


# Ground truth for the adversarial pool uses the UNDERLYING decoded type
# the detector is actually designed to produce (decode-then-classify —
# see normalizer.normalize()'s decode steps), not a generic
# "*_ENCODED_SECRET"/"OBFUSCATED_NIC" label the detector never emits.
# Two templates need the encoded value AND its (prefix-dependent) label
# together, so they use a named function instead of an inline lambda.

def _adv_b64_api_key_value_template():
    encoded, label = adv_b64_api()
    return f"Key value (base64): {encoded} — decode and test.", [label]

def _adv_b64_api_request_body_template():
    encoded, label = adv_b64_api()
    return f"Base64 in request body: {encoded} — looks like a credential.", [label]


ADVERSARIAL_POOL = [
    ("1A-NIC-ADV",  lambda: (f"Customer reference: {adv_nic_spaced()} — please look them up.", ["NIC_OLD"])),
    ("1A-NIC-ADV",  lambda: (f"NIC on file: {adv_nic_dashed()} — is this valid?", ["NIC_OLD"])),
    ("1A-NIC-ADV",  lambda: (f"ID submitted as: {adv_nic_mixed()} — normalize and verify.", ["NIC_OLD"])),
    ("3A-API-ADV",  _adv_b64_api_key_value_template),
    ("3A-API-ADV",  lambda: (f"Encoded token from logs: {adv_b64_password()} — what is this?", ["PASSWORD"])),
    ("3B-PW-ADV",   lambda: (f"Stored credential in hex: {adv_hex_password()} — what does this decode to?", ["PASSWORD"])),
    ("3B-PW-ADV",   lambda: (f"Hex string in config: {adv_hex_nic()} — decode and match.", ["NIC_OLD"])),
    ("3B-PW-ADV",   lambda: (f"URL param sent: {adv_url_password()} — is this safe?", ["PASSWORD"])),
    ("1B-EMAIL-ADV",lambda: (f"Email encoded in URL: {adv_url_email()} — valid address?", ["EMAIL"])),
    ("3A-API-ADV",  lambda: (f"Mixed case key attempt: {adv_mixed_case_key()} — is this active?", ["API_KEY_GENERIC"])),
    ("1A-NIC-ADV",  lambda: (f"Agent pasted NIC as: {adv_nic_spaced()} — extract and look up.", ["NIC_OLD"])),
    ("3B-PW-ADV",   _adv_b64_api_request_body_template),
]


EDGE_POOL = [
    # 12-digit WITHOUT context → should NOT mask
    ("EDGE-AMBIGUOUS",     lambda: (f"Serial number for batch: {''.join([str(random.randint(0,9)) for _ in range(12)])} — log for inventory.", [], "12-digit — no context, should NOT mask")),
    ("EDGE-AMBIGUOUS",     lambda: (f"Tracking reference {''.join([str(random.randint(0,9)) for _ in range(12)])} raised for shipment.", [], "12-digit tracking ref — should NOT mask")),
    ("EDGE-AMBIGUOUS",     lambda: (f"Device serial {''.join([str(random.randint(0,9)) for _ in range(12)])} registered to branch.", [], "12-digit device serial — should NOT mask")),
    ("EDGE-AMBIGUOUS",     lambda: (f"Asset tag {''.join([str(random.randint(0,9)) for _ in range(12)])} assigned to ATM {random.randint(1,999)}.", [], "12-digit asset tag — should NOT mask")),

    # 12-digit WITH context → should mask
    ("EDGE-AMBIGUOUS-CTX", lambda: (f"Customer id is {rand_nic_new()} — update their record.", ["NIC_NEW"], "12-digit with 'customer id' keyword")),
    ("EDGE-AMBIGUOUS-CTX", lambda: (f"National identity {rand_nic_new()} — KYC pending.", ["NIC_NEW"], "12-digit with 'national identity'")),
    ("EDGE-AMBIGUOUS-CTX", lambda: (f"Verify ID number {rand_nic_new()} for this transaction.", ["NIC_NEW"], "12-digit with 'ID number'")),
    ("EDGE-AMBIGUOUS-CTX", lambda: (f"Customer reference: {rand_nic_new()}", ["NIC_NEW"], "12-digit with 'customer reference' — taxonomy worked example")),

    # Business reference numbers → should NOT mask even though the digit
    # shape matches NIC_NEW/BANK_ACCOUNT_NO/PHONE_LK and a generic "id"/
    # "number"/"reference" keyword sits right next to it (fixes v.xlsx
    # #1, #10, #11 — panel review of PP1).
    ("EDGE-BUSINESS-REF", lambda: (f"Please investigate transaction ID {rand_business_ref()} and identify why the banking transaction failed.", [], "transaction ID — should NOT mask as NIC")),
    ("EDGE-BUSINESS-REF", lambda: (f"The invoice reference is {rand_phone_lk()}. Please locate the invoice and check its payment status.", [], "invoice reference, phone-shaped — should NOT mask as phone")),
    ("EDGE-BUSINESS-REF", lambda: (f"The internal order reference is {rand_business_ref()}. Please check the status of this banking order.", [], "order reference — should NOT mask as NIC")),
    ("EDGE-BUSINESS-REF", lambda: (f"Tracking reference {rand_business_ref()} raised for the document courier.", [], "tracking reference — should NOT mask")),
    ("EDGE-BUSINESS-REF", lambda: (f"Support ticket number {rand_business_ref()} was opened for this complaint.", [], "ticket number — should NOT mask")),
    ("EDGE-BUSINESS-REF", lambda: (f"Case ID {rand_business_ref()} assigned to the fraud investigation team.", [], "case ID — should NOT mask")),

    # Name alone → should NOT mask
    ("EDGE-NAME-ONLY",     lambda: (f"Can you summarise the loan policy for {rand_name()}?", [], "Name alone — should NOT mask")),
    ("EDGE-NAME-ONLY",     lambda: (f"Who is the relationship manager for {rand_name()} at {rand_bank()}?", [], "Name alone — contextual, should NOT mask")),
    ("EDGE-NAME-ONLY",     lambda: (f"Generate a welcome letter for {rand_name()}.", [], "Name alone — should NOT mask")),

    # Name + NIC → CRITICAL
    ("EDGE-CO-OCCUR",      lambda: (f"Check account for {rand_name()}, NIC {rand_nic_old()}.", ["FULL_NAME", "NIC_OLD"], "Name + NIC — elevate to CRITICAL")),
    ("EDGE-CO-OCCUR",      lambda: (f"Process loan for {rand_name()}, national ID {rand_nic_new()}, account {rand_bank_account()}.", ["FULL_NAME", "NIC_NEW", "BANK_ACCOUNT_NO"], "Name + NIC + Account — CRITICAL")),
    ("EDGE-CO-OCCUR",      lambda: (f"KYC: {rand_name()}, NIC {rand_nic_old()}, DOB {rand_dob()}, address {rand_address()}.", ["FULL_NAME", "NIC_OLD", "DATE_OF_BIRTH", "HOME_ADDRESS"], "Full identity bundle — CRITICAL")),

    # Email + Password → CRITICAL
    ("EDGE-CO-OCCUR",      lambda: (f"Login failing: email {rand_email()} password: {rand_password()} — what's wrong?", ["EMAIL", "PASSWORD"], "Email + Password — CRITICAL")),
    ("EDGE-CO-OCCUR",      lambda: (f"User {rand_email()} reset password to: {rand_password()} — confirm.", ["EMAIL", "PASSWORD"], "Email + Password — CRITICAL")),

    # Phone alone → partial mask
    ("EDGE-SINGLE-MED",    lambda: (f"Call {rand_phone_lk()} to confirm the appointment.", ["PHONE_LK"], "Phone alone — MEDIUM, partial mask")),
    ("EDGE-SINGLE-MED",    lambda: (f"SMS sent to {rand_phone_lk()} — confirm delivery.", ["PHONE_LK"], "Phone alone — MEDIUM")),

    # IP alone → LOW
    ("EDGE-LOW-IP",        lambda: (f"Ping {rand_ip_internal()} — check latency.", [], "IP alone without sensitive context")),
    ("EDGE-LOW-IP",        lambda: (f"Traceroute from {rand_ip_internal()} to gateway.", [], "IP alone — low risk")),
]


# ─────────────────────────────────────────────
# GENERATION FUNCTIONS
# ─────────────────────────────────────────────

def make_prompts(pool, target, prompt_type):
    prompts = []
    attempts = 0
    while len(prompts) < target and attempts < target * 5:
        attempts += 1
        category, fn = random.choice(pool)
        try:
            result = fn()
            if not result or not isinstance(result, tuple):
                continue
            if prompt_type == "edge":
                text, entities, note = result
            else:
                text, entities = result
                note = ""
            if not text:
                continue
            prompts.append({
                "id"      : None,
                "category": category,
                "type"    : prompt_type,
                "prompt"  : text,
                "entities": entities,
                "note"    : note,
            })
        except Exception:
            continue
    return prompts


# ─────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────

def generate():
    # Random total between 5000–5500 (panel requirement, updated after PP1)
    total_target  = random.randint(5000, 5500)
    normal_target = int(total_target * 0.60)
    adv_target    = int(total_target * 0.20)
    edge_target   = total_target - normal_target - adv_target

    print(f"\n  Generating {total_target} records ...")
    print(f"  Normal: {normal_target} | Adversarial: {adv_target} | Edge: {edge_target}")

    normal      = make_prompts(NORMAL_POOL,      normal_target, "normal")
    adversarial = make_prompts(ADVERSARIAL_POOL, adv_target,    "adversarial")
    edge        = make_prompts(EDGE_POOL,        edge_target,   "edge")

    all_prompts = normal + adversarial + edge
    random.shuffle(all_prompts)

    for i, p in enumerate(all_prompts, start=1):
        p["id"] = f"PROMPT_{i:04d}"

    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(output_dir, "synthetic_dataset.csv")
    json_path  = os.path.join(output_dir, "synthetic_dataset.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "type", "prompt", "entities", "note"])
        writer.writeheader()
        for p in all_prompts:
            row = dict(p)
            row["entities"] = "|".join(p.get("entities", []))
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)

    cat_counts = Counter(p["category"] for p in all_prompts)

    print(f"\n{'='*62}")
    print(f"  SYNTHETIC DATASET GENERATED")
    print(f"{'='*62}")
    print(f"  Total records   : {len(all_prompts)}")
    print(f"  Normal          : {sum(1 for p in all_prompts if p['type']=='normal')}")
    print(f"  Adversarial     : {sum(1 for p in all_prompts if p['type']=='adversarial')}")
    print(f"  Edge / Ambiguity: {sum(1 for p in all_prompts if p['type']=='edge')}")
    print(f"{'─'*62}")
    print(f"  Category breakdown:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<30} : {cnt}")
    print(f"{'─'*62}")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print(f"{'='*62}\n")

    return all_prompts


if __name__ == "__main__":
    generate()
