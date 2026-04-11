"""
ICFAI University Hyderabad — B.Tech Helpdesk Chatbot (PRO VERSION)
===================================================================
Author     : Upgraded from original by Y. Venkateswar Rao (23STUCHH010540)
Guide      : Dr. Dhanikonda Srinivasa Rao
Department : CSE (AI & ML), ICFAI University Hyderabad (IFHE)
Year       : 2024-2025

Features:
  ✅ Login / Logout system (session_state)
  ✅ Modern professional UI (gradient + rounded bubbles)
  ✅ Voice Input  — Browser Web Speech API  (🎤 Speak button)
  ✅ Voice Output — Browser Speech Synthesis (TTS on bot replies)
  ✅ ICFAI University, Hyderabad — B.Tech focused knowledge base
  ✅ TF-IDF + Cosine Similarity NLP (original logic preserved)
  ✅ Clear Chat, Suggested Questions, Better Error Handling
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import streamlit as st
import streamlit.components.v1 as components
import nltk
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# NLTK data download (runs once)
# ──────────────────────────────────────────────
for _resource in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.data.find(
            f"tokenizers/{_resource}" if "punkt" in _resource else f"corpora/{_resource}"
        )
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))

# ──────────────────────────────────────────────
# AUTH CONFIG  (extend with a real DB in prod)
# ──────────────────────────────────────────────
VALID_USERS = {
    "venky": "1234",
    "student": "ifhe2025",   # bonus test account
}

# ──────────────────────────────────────────────
# KNOWLEDGE BASE  — ICFAI University Hyderabad
# ──────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # ── GREETINGS ──────────────────────────────
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "what's up", "greetings", "hiya",
        ],
        "responses": [
            "Hey there! 👋 Welcome to the ICFAI University Helpdesk. How can I assist you today?",
            "Hello! I'm IFHE Bot — your 24/7 B.Tech guide at ICFAI University, Hyderabad. Ask me anything! 😊",
            "Hi! Great to see you. Ask me about fees, exams, hostel, placements, or anything B.Tech related!",
            "Greetings! I'm your ICFAI University assistant. What can I help you with today?",
        ],
    },
    # ── FAREWELL ───────────────────────────────
    {
        "tag": "farewell",
        "patterns": [
            "bye", "goodbye", "see you", "take care", "exit", "quit",
            "that's all", "thanks bye", "cya", "later",
        ],
        "responses": [
            "Take care! Come back whenever you have more questions. 😊 — ICFAI Helpdesk",
            "Goodbye! Best of luck with your B.Tech studies at IFHE! 🎓",
            "See you later! Feel free to return if you need anything else. 👋",
            "Bye! Have a great day on campus! ☀️",
        ],
    },
    # ── THANKS ─────────────────────────────────
    {
        "tag": "thanks",
        "patterns": [
            "thanks", "thank you", "thank you so much", "many thanks",
            "that's helpful", "that helped", "got it thanks", "cheers",
        ],
        "responses": [
            "You're welcome! 😊 Anything else I can help with at ICFAI?",
            "Happy to help! Let me know if there's anything else you need.",
            "No problem at all! That's what I'm here for — IFHE Bot 🤖",
            "Glad I could help! Feel free to ask more questions anytime.",
        ],
    },
    # ── FEES ───────────────────────────────────
    {
        "tag": "fees",
        "patterns": [
            "what is the fee structure",
            "how much is the tuition fee",
            "fee payment deadline",
            "when is the last date to pay fees",
            "annual fees",
            "semester fee amount",
            "how to pay college fees",
            "fee concession",
            "fee installment option",
            "late fee penalty",
            "hostel fee",
            "bus fee transport fee",
            "icfai fee",
            "ifhe fee",
        ],
        "responses": [
            (
                "💰 **Fee Structure — B.Tech at ICFAI University, Hyderabad:**\n\n"
                "• **Annual Tuition Fee:** ≈ ₹1,20,000 per year (varies by branch)\n"
                "• **Hostel Fee:** ₹90,000 – ₹1,10,000 per year (AC/Non-AC; includes mess)\n"
                "• **Bus/Transport Fee:** ₹15,000 – ₹22,000 (depends on route)\n"
                "• **One-time Admission Fee:** ₹25,000 (paid at the time of joining)\n\n"
                "📅 Fees are paid **semester-wise** (two instalments/year). The due date is "
                "typically within **30 days** of semester commencement.\n\n"
                "💳 **Payment Modes:** ICFAI ERP portal (online), NEFT/RTGS, or at the "
                "Accounts Department (Admin Block, Ground Floor).\n\n"
                "⚠️ A late fee of ₹100/day is charged after the deadline."
            ),
            (
                "💳 **Fee Payment — ICFAI University Hyderabad:**\n\n"
                "Log in to the **ICFAI ERP (erp.ifheindia.org)** → Finance → Fee Payment.\n\n"
                "For **fee waivers or installment requests**, submit an application at the "
                "Finance Office (Admin Block) with your parent's income proof.\n\n"
                "**Scholarships** (SC/ST/BC/EBC/OBC) can substantially reduce the fee — "
                "apply via the Telangana e-Pass portal and submit a copy to the Scholarship Cell."
            ),
        ],
    },
    # ── EXAMS ──────────────────────────────────
    {
        "tag": "exams",
        "patterns": [
            "when are semester exams",
            "exam schedule",
            "examination dates",
            "mid exams",
            "internal assessment",
            "exam timetable",
            "hall ticket",
            "admit card",
            "exam results",
            "supplementary exams",
            "how to apply for recounting",
            "exam fee",
            "malpractice rules",
            "icfai exam",
            "ifhe exam",
        ],
        "responses": [
            (
                "📝 **Examination Schedule — ICFAI University Hyderabad:**\n\n"
                "• **Odd Semester (Sem 1, 3, 5, 7) Exams:** November – December\n"
                "• **Even Semester (Sem 2, 4, 6, 8) Exams:** April – May\n"
                "• **Mid-Term Tests (CIA):** Two per semester (approx. Week 7 & Week 14)\n\n"
                "📋 Hall tickets are available on the **ICFAI ERP portal** 10–14 days before exams.\n\n"
                "📊 Results are published on the **ICFAI results portal** within 4–6 weeks. "
                "You may apply for **re-evaluation** within 15 days of result publication "
                "(fee: ₹500/subject)."
            ),
            (
                "🎓 **Exam Rules — ICFAI University:**\n\n"
                "• Carry hall ticket **+** college ID to every exam\n"
                "• Entry allowed up to **30 minutes** after exam start time only\n"
                "• Mobile phones / smartwatches are **strictly prohibited** in the exam hall\n"
                "• Malpractice → case referred to the University Disciplinary Committee\n"
                "• For hall ticket errors, contact the Exam Section (Admin Block, Room 110) "
                "immediately — bring your ERP screenshot."
            ),
        ],
    },
    # ── HOSTEL ─────────────────────────────────
    {
        "tag": "hostel",
        "patterns": [
            "how to apply for hostel",
            "hostel allotment",
            "hostel availability",
            "is hostel available",
            "hostel rules",
            "hostel room",
            "hostel facilities",
            "hostel warden",
            "hostel in time",
            "hostel mess",
            "hostel fee payment",
            "leave from hostel",
            "icfai hostel",
            "ifhe hostel",
        ],
        "responses": [
            (
                "🏠 **Hostel Information — ICFAI University Hyderabad:**\n\n"
                "• Separate hostels for **boys** and **girls** on the Donthanapally campus\n"
                "• Apply online through the **ICFAI ERP** → Hostel Module at the start of "
                "each academic year\n"
                "• **Room Types:** Double/Triple sharing (Non-AC) | Single/Double (AC)\n\n"
                "🍽️ **Facilities:** 24×7 Wi-Fi, CCTV security, mess (3 meals + evening snacks), "
                "laundry service, indoor games room, common TV lounge\n\n"
                "⏰ **In-time:** 9:30 PM (weekdays) | 10:30 PM (weekends & holidays)\n"
                "🧑‍💼 For warden contact, visit the Hostel Office near the main gate or call "
                "the ICFAI helpline: **040-2345-6789**"
            ),
        ],
    },
    # ── PLACEMENT ──────────────────────────────
    {
        "tag": "placement",
        "patterns": [
            "placement opportunities",
            "campus placements",
            "when do companies visit",
            "placement training",
            "how to register for placements",
            "highest package",
            "average salary",
            "which companies recruit",
            "placement cell contact",
            "internship opportunities",
            "off campus placement",
            "placement statistics",
            "icfai placement",
            "ifhe placement",
        ],
        "responses": [
            (
                "💼 **Placement & Internship — ICFAI University Hyderabad:**\n\n"
                "• The **Centre for Career Development (CCD)** manages all placements\n"
                "• Regular recruiters: **TCS, Infosys, Wipro, Capgemini, Accenture, "
                "Amazon, Deloitte, KPMG, HCL, Cognizant**, and 100+ more\n\n"
                "📅 **Timeline:**\n"
                "• Summer Internships: Feb – March registration (3rd year)\n"
                "• Final Placements: July – December (4th year)\n\n"
                "📝 **Register:** ERP → Placement Portal → Complete your profile & upload résumé\n\n"
                "📊 **2023-24 Highlights:**\n"
                "• Placement rate: **~90%** (B.Tech)\n"
                "• Highest Package: **₹28 LPA** (CSE branch)\n"
                "• Average Package: **₹6.5 LPA**\n"
                "• 300+ companies visited campus"
            ),
        ],
    },
    # ── LIBRARY ────────────────────────────────
    {
        "tag": "library",
        "patterns": [
            "library timings",
            "library hours",
            "library books",
            "how to borrow books",
            "library membership",
            "issue books",
            "return books",
            "library fine",
            "digital library",
            "e-books access",
            "library facilities",
            "library rules",
        ],
        "responses": [
            (
                "📚 **Library — ICFAI University Hyderabad:**\n\n"
                "• **Timings:** Monday – Saturday, 8:00 AM – 9:00 PM\n"
                "• Exam season extended hours: till 10:00 PM\n\n"
                "📖 **Borrowing Rules:**\n"
                "• B.Tech students: Up to **4 books** for **14 days**\n"
                "• Late return fine: ₹2 per book per day\n\n"
                "💻 **Digital Resources:**\n"
                "• **DELNET, IEEE Xplore, Springer, EBSCO** (use ICFAI login on campus/VPN)\n"
                "• NPTEL video lectures also accessible via the Library Portal\n\n"
                "🪪 College ID is mandatory. Lost books → pay replacement cost + ₹100 penalty."
            ),
        ],
    },
    # ── ATTENDANCE ─────────────────────────────
    {
        "tag": "attendance",
        "patterns": [
            "what is the attendance requirement",
            "minimum attendance",
            "attendance percentage",
            "how much attendance needed",
            "attendance shortage",
            "attendance condonation",
            "attendance rules",
            "how to check attendance",
            "attendance for exam",
            "detained due to attendance",
        ],
        "responses": [
            (
                "📊 **Attendance Policy — ICFAI University Hyderabad:**\n\n"
                "• **Minimum Required:** **75%** in every subject\n"
                "• Below 75% → **Detained** from semester exams (no exceptions without condonation)\n\n"
                "📋 **Condonation (65–74%):** Apply to the HOD/Dean with a valid medical certificate "
                "or official documentation. Approved on a case-by-case basis.\n\n"
                "📱 **Check Attendance:** ERP → Academics → Attendance Report (updated every week)\n\n"
                "⚠️ Tip: Keep a buffer — target **≥ 85%** so a medical emergency doesn't push "
                "you below the cutoff. Talk to your Faculty Advisor for any discrepancies ASAP."
            ),
        ],
    },
    # ── ADMISSIONS ─────────────────────────────
    {
        "tag": "admissions",
        "patterns": [
            "admission process",
            "how to take admission",
            "eligibility for btech",
            "entrance exam for admission",
            "eamcet",
            "jee admission",
            "lateral entry admission",
            "admission last date",
            "documents for admission",
            "scholarship for admission",
            "icfai admission",
            "ifhe admission",
        ],
        "responses": [
            (
                "🎓 **Admission Process — B.Tech at ICFAI University Hyderabad:**\n\n"
                "**1st Year (Direct Entry):**\n"
                "• Via **ICFAI Admission Test (IAT)** or JEE Main score\n"
                "• Eligibility: 10+2 with PCM, minimum **60%** aggregate\n\n"
                "**2nd Year (Lateral Entry):**\n"
                "• Diploma holders (3-year Poly) or B.Sc (PCM) graduates\n\n"
                "📄 **Documents Required:**\n"
                "10th & 12th marksheets, TC, Character Certificate, "
                "Community Certificate, Aadhar, 4 passport photos, Migration Certificate\n\n"
                "📅 Admission cycle: **May – August** each year.\n"
                "🌐 Apply online: **www.ifheindia.org/admissions**\n"
                "📞 Admissions helpline: **1800-XXX-XXXX** (toll-free, 9AM–5PM)"
            ),
        ],
    },
    # ── SCHOLARSHIPS ───────────────────────────
    {
        "tag": "scholarships",
        "patterns": [
            "scholarship information",
            "fee reimbursement",
            "ebc scholarship",
            "bc scholarship",
            "sc st scholarship",
            "merit scholarship",
            "how to apply for scholarship",
            "scholarship eligibility",
            "post matric scholarship",
            "scholarship amount",
        ],
        "responses": [
            (
                "🏆 **Scholarships — ICFAI University Hyderabad:**\n\n"
                "• **Telangana State Fee Reimbursement:** SC/ST/BC/EBC categories "
                "(family income ≤ ₹2.5 lakh/year). Apply via **telanganaepass.cgg.gov.in**\n\n"
                "• **ICFAI Merit Scholarship:** Top 5% students per branch get 25–50% tuition waiver\n\n"
                "• **ICFAI INSPIRE Scholarship:** For students with JEE Main rank ≤ 50,000\n\n"
                "• **Sports/Cultural Scholarship:** Contact Dean of Student Affairs\n\n"
                "📅 Applications open every **August – October**.\n"
                "📍 Scholarship Cell: Admin Block, Room 108 | scholarship@ifheindia.org"
            ),
        ],
    },
    # ── CLUBS & ACTIVITIES ──────────────────────
    {
        "tag": "clubs_activities",
        "patterns": [
            "college clubs",
            "extracurricular activities",
            "technical club",
            "cultural events",
            "sports activities",
            "how to join club",
            "annual day",
            "techfest",
            "cultural fest",
            "student council",
            "nss ncc",
        ],
        "responses": [
            (
                "🎭 **Clubs & Activities — ICFAI University Hyderabad:**\n\n"
                "• **Technical:** ICFAI Coding Club, IEEE Student Branch, Robotics Club, "
                "AI/ML Society, Cybersecurity Club\n"
                "• **Cultural:** Rhythm (Dance), Resonance (Music), Drama Guild, Fine Arts\n"
                "• **Sports:** Cricket, Football, Basketball, Badminton, Chess, Athletics\n"
                "• **Social:** NSS Unit, Rotaract Club, Blood Donation Cell\n\n"
                "📅 Club registrations open in **August** (1st week of Odd Semester).\n"
                "🎉 **VERVE** (Annual Cultural Fest) & **TechNITROUS** (Tech Fest) — every February!\n\n"
                "Follow **@icfaiuniversity.ifhe** on Instagram for updates."
            ),
        ],
    },
    # ── TRANSPORT ──────────────────────────────
    {
        "tag": "transport",
        "patterns": [
            "college bus",
            "bus routes",
            "transport facilities",
            "bus timings",
            "bus pass",
            "how to get bus pass",
            "which areas does bus cover",
            "bus schedule",
        ],
        "responses": [
            (
                "🚌 **Transport — ICFAI University Hyderabad:**\n\n"
                "• **30+ bus routes** covering Hyderabad, Secunderabad, Uppal, LB Nagar, "
                "Kukatpally, Dilsukhnagar, Mehdipatnam, and surrounding areas\n"
                "• **Morning departure:** Pick-up points between **7:00 – 8:00 AM**\n"
                "• **Evening return:** Leaves campus at **5:15 PM** (Mon–Fri) & **1:15 PM** (Sat)\n\n"
                "🎫 **Bus Pass:** Apply at the Transport Office (Admin Block, Room 002) "
                "at the start of each semester.\n"
                "• Annual fee: ₹15,000 – ₹22,000 (based on distance)\n\n"
                "📍 For full route schedules, check the ICFAI ERP → Transport section."
            ),
        ],
    },
    # ── CERTIFICATES ───────────────────────────
    {
        "tag": "certificates",
        "patterns": [
            "bonafide certificate",
            "transfer certificate",
            "tc",
            "no objection certificate",
            "noc",
            "provisional certificate",
            "degree certificate",
            "migration certificate",
            "how to get character certificate",
            "certificate from college",
        ],
        "responses": [
            (
                "📜 **Certificates & Documents — ICFAI University Hyderabad:**\n\n"
                "| Certificate | Apply At | Processing Time |\n"
                "|---|---|---|\n"
                "| Bonafide Certificate | Academic Section | 1–2 working days |\n"
                "| Character Certificate | Academic Section | 2–3 working days |\n"
                "| Transfer Certificate | Dean's Office | 5–7 working days |\n"
                "| NOC (Visa/Job) | Registrar's Office | 3–5 working days |\n"
                "| Provisional Certificate | Exam Branch | 7–10 working days |\n"
                "| Migration Certificate | Registrar's Office | 7–10 working days |\n\n"
                "📝 Submit request on the **ICFAI ERP** (Student Services tab) or visit the "
                "respective office with your Student ID. Fee: ₹50–₹200 depending on document."
            ),
        ],
    },
    # ── WI-FI ──────────────────────────────────
    {
        "tag": "wifi_internet",
        "patterns": [
            "college wifi",
            "internet access",
            "wifi password",
            "how to connect wifi",
            "campus internet",
            "broadband in hostel",
            "wifi not working",
        ],
        "responses": [
            (
                "📶 **Wi-Fi & Internet — ICFAI University Hyderabad:**\n\n"
                "• **Campus Wi-Fi (SSID: IFHE-Campus):** Available in all academic blocks, "
                "library, canteen, and hostel\n"
                "• Login: Use your **ERP username & password** (portal.ifheindia.org login)\n"
                "• Speed: Up to **1 Gbps** backbone (shared); typical student speed ~50 Mbps\n\n"
                "🔧 **Issues?** Contact the **IT Help Desk:**\n"
                "• Location: Computer Centre, Room 002\n"
                "• Email: itsupport@ifheindia.org\n"
                "• ERP → IT Support → Raise Ticket"
            ),
        ],
    },
    # ── CONTACT INFO ───────────────────────────
    {
        "tag": "contact_info",
        "patterns": [
            "college phone number",
            "contact details",
            "principal contact",
            "department contact",
            "how to contact college",
            "email id of college",
            "address of college",
            "helpline number",
            "who to contact",
            "icfai contact",
            "ifhe phone",
        ],
        "responses": [
            (
                "📞 **Contact — ICFAI University Hyderabad (IFHE):**\n\n"
                "🏫 **Address:** ICFAI University, Donthanapally, Shankarapalli Road, "
                "Hyderabad – 501203, Telangana, India\n\n"
                "📞 **Main Reception:** +91-40-2304-5000\n"
                "📱 **Student Helpline:** 1800-XXX-XXXX (Toll-Free, 9AM–5PM)\n\n"
                "📧 **Key Emails:**\n"
                "• General: info@ifheindia.org\n"
                "• Admissions: admissions@ifheindia.org\n"
                "• Placements: placements@ifheindia.org\n"
                "• Exams: examcell@ifheindia.org\n\n"
                "⏰ Office Hours: Mon–Sat, 9:00 AM – 5:00 PM\n"
                "🌐 **Website:** www.ifheindia.org"
            ),
        ],
    },
    # ── ABOUT ICFAI ────────────────────────────
    {
        "tag": "about_college",
        "patterns": [
            "about this college",
            "tell me about the college",
            "history of the college",
            "how old is the college",
            "is it autonomous",
            "which university affiliated",
            "naac accreditation",
            "nba accredited",
            "college ranking",
            "about icfai",
            "about ifhe",
            "icfai university hyderabad",
        ],
        "responses": [
            (
                "🏛️ **About ICFAI University Hyderabad (IFHE):**\n\n"
                "• Established: **2004** | UGC-recognized Private University\n"
                "• Located at Donthanapally, Shankarapalli Road, Hyderabad — a 140-acre lush campus\n"
                "• **Autonomous** since inception — offers its own degrees (not affiliated to JNTUH)\n\n"
                "🏆 **Accreditations & Rankings:**\n"
                "• **NAAC 'A+' Grade** accredited\n"
                "• **NBA Accredited** programs: B.Tech CSE, ECE, EEE, Mechanical\n"
                "• Consistently ranked among **Top 75 Engineering Universities** in India (NIRF)\n\n"
                "🎓 10,000+ students across UG, PG, and PhD programs. "
                "Strong focus on research, industry partnerships, and entrepreneurship."
            ),
        ],
    },
    # ── BRANCHES / COURSES ─────────────────────
    {
        "tag": "branches",
        "patterns": [
            "which branches are available",
            "btech courses offered",
            "cse branch",
            "ece branch",
            "mechanical branch",
            "civil branch",
            "data science course",
            "artificial intelligence branch",
            "available specializations",
        ],
        "responses": [
            (
                "🖥️ **B.Tech Branches at ICFAI University Hyderabad:**\n\n"
                "• **CSE** — Computer Science & Engineering\n"
                "• **CSE (AI & ML)** — Artificial Intelligence & Machine Learning\n"
                "• **CSE (Data Science)** — Data Science & Analytics\n"
                "• **CSE (Cyber Security)** — Cybersecurity\n"
                "• **ECE** — Electronics & Communication Engineering\n"
                "• **EEE** — Electrical & Electronics Engineering\n"
                "• **Mechanical Engineering**\n"
                "• **Civil Engineering**\n\n"
                "All branches are 4-year (8-semester) full-time programs. "
                "CSE-AI/ML and Data Science are the most sought-after branches at IFHE."
            ),
        ],
    },
    # ── ERP / PORTAL ───────────────────────────
    {
        "tag": "erp_portal",
        "patterns": [
            "how to use erp",
            "erp login",
            "student portal",
            "icfai portal",
            "forgot erp password",
            "how to access marks",
            "online portal",
            "ifhe erp",
        ],
        "responses": [
            (
                "🖥️ **ICFAI ERP / Student Portal:**\n\n"
                "• **URL:** https://erp.ifheindia.org\n"
                "• **Login:** Use your Student ID (Roll Number) as username; "
                "default password is given during orientation (change it immediately!)\n\n"
                "📌 **What you can do on ERP:**\n"
                "→ View attendance, marks, fee dues\n"
                "→ Download hall ticket & results\n"
                "→ Apply for bonafide/certificates\n"
                "→ Register for placements & internships\n"
                "→ Pay fees online\n\n"
                "🔑 **Forgot Password?** Click 'Forgot Password' on the login page or "
                "visit the IT Help Desk (Computer Centre, Room 002)."
            ),
        ],
    },
]

# ──────────────────────────────────────────────
# NLP UTILITIES  (original logic — preserved)
# ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def get_all_patterns() -> list:
    """Return (preprocessed_pattern, intent_index) for every pattern."""
    result = []
    for idx, intent in enumerate(KNOWLEDGE_BASE):
        for pattern in intent["patterns"]:
            result.append((preprocess(pattern), idx))
    return result


# Pre-build vectorizer at module load time
_pattern_data = get_all_patterns()
_pattern_texts = [p[0] for p in _pattern_data]
_pattern_intent_indices = [p[1] for p in _pattern_data]
_vectorizer = TfidfVectorizer()
_tfidf_matrix = _vectorizer.fit_transform(_pattern_texts)


def find_response(user_input: str, threshold: float = 0.20) -> str:
    """
    Match user query to the best intent via TF-IDF cosine similarity.
    Falls back to a helpful 'not sure' message if confidence is too low.
    """
    cleaned = preprocess(user_input)
    if not cleaned:
        return "I didn't quite catch that. Could you rephrase? 😊"

    lower = user_input.lower().strip()

    # Fast-path for greetings / farewells / thanks
    for intent in KNOWLEDGE_BASE:
        if intent["tag"] in ("greeting", "farewell", "thanks"):
            if any(p in lower for p in intent["patterns"]):
                return random.choice(intent["responses"])

    try:
        user_vec = _vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vec, _tfidf_matrix).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            intent_idx = _pattern_intent_indices[best_idx]
            return random.choice(KNOWLEDGE_BASE[intent_idx]["responses"])
        else:
            return (
                "🤔 Hmm, I'm not entirely sure about that one. I'm best at helping with:\n\n"
                "💰 **Fees** | 📝 **Exams** | 🏠 **Hostel** | 💼 **Placements** | "
                "📚 **Library** | 📊 **Attendance** | 🎓 **Admissions** | 🏆 **Scholarships** | "
                "🚌 **Transport** | 📜 **Certificates** | 📶 **Wi-Fi** | 🖥️ **ERP Portal** | "
                "🏛️ **About ICFAI**\n\n"
                "Could you try rephrasing your question?"
            )
    except Exception:
        return "Something went wrong on my end. Please try again! 🔄"


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
def setup_page():
    st.set_page_config(
        page_title="ICFAI University Helpdesk",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded",
    )


# ──────────────────────────────────────────────
# CSS INJECTION  — Professional Deep-Blue Theme
# ──────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Poppins:wght@400;600;700&display=swap');

        /* ── Base ── */
        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #0d1b3e 0%, #0a2a5c 40%, #0e3d72 100%);
            min-height: 100vh;
        }

        /* ── Hide default streamlit chrome ── */
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }
        header     { visibility: hidden; }

        /* ── LOGIN CARD ── */
        .login-card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 24px;
            padding: 40px 36px 36px;
            max-width: 420px;
            margin: 60px auto 0;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            text-align: center;
        }
        .login-card h1 {
            font-family: 'Poppins', sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 4px;
        }
        .login-card p {
            color: rgba(255,255,255,0.65);
            font-size: 0.88rem;
            margin-bottom: 28px;
        }
        .login-logo { font-size: 3.2rem; margin-bottom: 10px; }

        /* ── HEADER BANNER ── */
        .chat-header {
            background: linear-gradient(135deg, #0077cc 0%, #00b4d8 100%);
            color: white;
            padding: 20px 28px;
            border-radius: 18px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,119,204,0.35);
            border: 1px solid rgba(255,255,255,0.15);
        }
        .chat-header h1 {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .chat-header p {
            margin: 5px 0 0;
            font-size: 0.85rem;
            opacity: 0.88;
        }
        .online-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.75rem;
            margin-top: 8px;
        }
        .online-badge::before {
            content: "●";
            color: #6effa0;
            margin-right: 5px;
        }

        /* ── CHAT BUBBLES ── */
        .chat-container { display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px; }

        .user-bubble-wrap { display: flex; justify-content: flex-end; align-items: flex-end; gap: 6px; }
        .bot-bubble-wrap  { display: flex; justify-content: flex-start; align-items: flex-end; gap: 6px; }

        .user-bubble {
            background: #1976d2;
            color: #ffffff !important;
            padding: 12px;
            border-radius: 15px;
            margin: 5px;
            font-size: 15px;
        }
       
        .bot-bubble {
            background: #f5f7fa;
            color: #111111 !important;
            padding: 12px;
            border-radius: 15px;
            margin: 5px;
            border-left: 5px solid #1976d2;
            font-size: 15px;
        }

        /* Ensure all child elements inside bot bubble stay dark */
        .bot-bubble * {
            color: #111111 !important;
        }
        .avatar {
            font-size: 1.5rem;
            flex-shrink: 0;
        }
        .timestamp {
            font-size: 0.68rem;
            opacity: 0.5;
            margin-top: 3px;
            text-align: right;
            color: rgba(255,255,255,0.7);
        }

        /* ── INPUT AREA ── */
        .stTextInput > div > div > input {
            border-radius: 30px !important;
            border: 2px solid rgba(0,180,216,0.5) !important;
            background: rgba(255,255,255,0.07) !important;
            color: white !important;
            padding: 12px 22px !important;
            font-size: 0.95rem !important;
            font-family: 'Nunito', sans-serif !important;
        }
        .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.45) !important; }
        .stTextInput > div > div > input:focus {
            border-color: #00b4d8 !important;
            box-shadow: 0 0 0 3px rgba(0,180,216,0.25) !important;
        }
        .stTextInput > label { display: none !important; }

        /* ── BUTTONS ── */
        .stButton > button {
            border-radius: 30px !important;
            font-family: 'Nunito', sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            border: none !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }
        div[data-testid="column"]:first-child .stButton > button {
            background: linear-gradient(135deg, #0077cc, #00b4d8) !important;
            color: white !important;
            padding: 10px 20px !important;
            box-shadow: 0 4px 14px rgba(0,119,204,0.4) !important;
        }
        div[data-testid="column"]:first-child .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(0,119,204,0.55) !important;
        }

        /* ── SIDEBAR ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1f4d 0%, #0d2a62 100%) !important;
            border-right: 1px solid rgba(0,180,216,0.2);
        }
        /* Only style specific text elements — avoids breaking interactive widgets */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p {
            color: rgba(255,255,255,0.9) !important;
        }
        .sidebar-topic {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(0,180,216,0.2);
            border-radius: 10px;
            padding: 8px 12px;
            margin-bottom: 7px;
            font-size: 0.83rem;
            transition: all 0.2s;
        }
        .sidebar-topic:hover {
            background: rgba(0,180,216,0.12);
            border-color: rgba(0,180,216,0.5);
        }
        .sidebar-topic b { font-size: 0.9rem; }
        .sidebar-topic i { color: rgba(255,255,255,0.55) !important; font-size: 0.8rem; }

        /* ── SUGGESTED QUESTIONS ── */
        .suggest-btn .stButton > button {
            background: rgba(255,255,255,0.07) !important;
            border: 1px solid rgba(0,180,216,0.35) !important;
            color: rgba(255,255,255,0.85) !important;
            font-size: 0.8rem !important;
            padding: 7px 14px !important;
            text-align: left !important;
            width: 100% !important;
        }
        .suggest-btn .stButton > button:hover {
            background: rgba(0,180,216,0.15) !important;
            border-color: #00b4d8 !important;
        }

        /* ── VOICE WIDGET ── */
        .voice-status {
            background: rgba(0,180,216,0.1);
            border: 1px solid rgba(0,180,216,0.3);
            border-radius: 10px;
            padding: 8px 14px;
            font-size: 0.82rem;
            color: rgba(255,255,255,0.8) !important;
            text-align: center;
            margin-top: 8px;
        }

        /* ── DIVIDER ── */
        hr { border-color: rgba(255,255,255,0.08) !important; }

        /* ── MISC ── */
        .stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.88); }

        /* Prevent global markdown colour from bleeding into bot chat bubbles */
        .bot-bubble p, .bot-bubble li {
            color: #111111 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# VOICE WIDGET  (Browser Web Speech API via JS)
# ──────────────────────────────────────────────
VOICE_COMPONENT_HTML = """
<style>
  #voice-btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, #0077cc, #00b4d8);
    color: white; border: none; border-radius: 30px;
    padding: 9px 20px; font-size: 14px; font-family: 'Nunito', sans-serif;
    font-weight: 700; cursor: pointer;
    box-shadow: 0 4px 14px rgba(0,119,204,0.4);
    transition: all 0.2s;
  }
  #voice-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,119,204,0.55); }
  #voice-btn.listening {
    background: linear-gradient(135deg, #cc0044, #ff4488);
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%,100%  { box-shadow: 0 4px 14px rgba(204,0,68,0.4); }
    50%      { box-shadow: 0 4px 28px rgba(204,0,68,0.7); }
  }
  #voice-status {
    font-size: 12px; color: rgba(255,255,255,0.65);
    font-family: 'Nunito', sans-serif; margin-top: 6px;
  }
</style>

<button id="voice-btn" onclick="startVoice()">🎤 Speak</button>
<div id="voice-status">Click to speak your question</div>

<script>
var recognition;
var supported = ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);

function startVoice() {
  if (!supported) {
    document.getElementById('voice-status').innerText = '❌ Speech not supported in this browser. Try Chrome.';
    return;
  }
  var btn = document.getElementById('voice-btn');
  var status = document.getElementById('voice-status');

  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'en-IN';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = function() {
    btn.classList.add('listening');
    btn.innerText = '🔴 Listening...';
    status.innerText = '🎙️ Speak now — I\'m listening...';
  };

  recognition.onresult = function(event) {
    var transcript = event.results[0][0].transcript;
    status.innerText = '✅ Heard: "' + transcript + '"';
    // Send recognised text to Streamlit via query param trick
    window.parent.postMessage({type: 'voice_input', text: transcript}, '*');
  };

  recognition.onerror = function(event) {
    btn.classList.remove('listening');
    btn.innerText = '🎤 Speak';
    if (event.error === 'not-allowed') {
      status.innerText = '❌ Microphone permission denied. Please allow mic access.';
    } else if (event.error === 'no-speech') {
      status.innerText = '⚠️ No speech detected. Try again!';
    } else {
      status.innerText = '❌ Error: ' + event.error;
    }
  };

  recognition.onend = function() {
    btn.classList.remove('listening');
    btn.innerText = '🎤 Speak';
  };

  recognition.start();
}

// Receive voice text back into the Streamlit input
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'voice_input') {
    // Try to fill the Streamlit text input
    var inputs = window.parent.document.querySelectorAll('input[type="text"]');
    inputs.forEach(function(inp) {
      var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      nativeInputValueSetter.call(inp, e.data.text);
      inp.dispatchEvent(new Event('input', { bubbles: true }));
    });
  }
});
</script>
"""

TTS_SCRIPT = """
<script>
  window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'tts_speak') {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        var utterance = new SpeechSynthesisUtterance(e.data.text);
        utterance.lang = 'en-IN';
        utterance.rate = 0.92;
        utterance.pitch = 1.05;
        window.speechSynthesis.speak(utterance);
      }
    }
  });
</script>
"""


def speak_text_js(text: str):
    """Trigger browser TTS for the given text (strips markdown)."""
    # Strip markdown symbols for cleaner speech
    clean = re.sub(r"[*_`#|>]", "", text)
    clean = re.sub(r"\n+", " ", clean).strip()
    clean = clean[:600]  # limit length
    components.html(
        f"""
        <script>
          var msg = new SpeechSynthesisUtterance({repr(clean)});
          msg.lang = 'en-IN';
          msg.rate = 0.92;
          msg.pitch = 1.05;
          if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
          }}
        </script>
        """,
        height=0,
    )


# ──────────────────────────────────────────────
# LOGIN PAGE
# ──────────────────────────────────────────────
def render_login():
    """Render the login page. Returns True if login successful."""
    st.markdown(
        """
        <div class="login-card">
          <div class="login-logo">🎓</div>
          <h1>ICFAI University Helpdesk</h1>
          <p>B.Tech Student Portal — Sign in to continue</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centre the form with columns
    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

        if st.button("🔐 Login to Helpdesk", use_container_width=True):
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

        st.markdown(
            "<div style='text-align:center; margin-top:14px; font-size:0.78rem; "
            "color:rgba(255,255,255,0.4);'>Demo: username <b>venky</b> / password <b>1234</b></div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # User greeting
        uname = st.session_state.get("username", "Student")
        st.markdown(
            f"""
            <div style='text-align:center; padding:16px 0 8px;'>
              <div style='font-size:2.4rem;'>👨‍🎓</div>
              <div style='font-size:1rem; font-weight:700;'>Hey, {uname.capitalize()}!</div>
              <div style='font-size:0.75rem; opacity:0.55;'>B.Tech @ ICFAI Hyderabad</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**📌 Quick Topic Guide**", unsafe_allow_html=True)

        topics = {
            "💰 Fees": "fee structure, payment, installments",
            "📝 Exams": "schedule, hall ticket, results",
            "🏠 Hostel": "allotment, facilities, rules",
            "💼 Placements": "CCD, package, internships",
            "📚 Library": "timings, borrowing, digital",
            "📊 Attendance": "75% rule, condonation",
            "🎓 Admissions": "IAT, JEE, lateral entry",
            "🏆 Scholarships": "e-Pass, ICFAI merit award",
            "🚌 Transport": "bus routes, pass, timings",
            "📜 Certificates": "bonafide, TC, NOC",
            "🖥️ ERP Portal": "login, marks, fee payment",
            "📶 Wi-Fi": "IFHE-Campus network, IT support",
            "🏛️ About ICFAI": "rankings, accreditations",
        }

        for label, hint in topics.items():
            st.markdown(
                f'<div class="sidebar-topic"><b>{label}</b><br><i>{hint}</i></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**⚙️ Settings**")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [_welcome_message()]
            st.rerun()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style='font-size:0.72rem; opacity:0.45; text-align:center; line-height:1.7;'>
            ICFAI University Helpdesk Bot<br>
            B.Tech CSE (AI &amp; ML) Project<br>
            IFHE, Hyderabad · 2024-25
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# CHAT RENDER
# ──────────────────────────────────────────────
def _welcome_message() -> dict:
    return {
        "role": "bot",
        "text": (
            "👋 Welcome to **ICFAI University Hyderabad Helpdesk**!\n\n"
            "I'm **IFHE Bot** — your 24/7 AI guide for all B.Tech related queries.\n\n"
            "Ask me about 💰 **Fees** • 📝 **Exams** • 🏠 **Hostel** • 💼 **Placements** • "
            "📚 **Library** • 📊 **Attendance** • 🎓 **Admissions** • 🏆 **Scholarships** "
            "• 🚌 **Transport** • 🖥️ **ERP Portal** and much more!\n\n"
            "What would you like to know? 😊"
        ),
    }


def render_chat():
    """Render conversation history as styled chat bubbles."""
    msgs = st.session_state.get("messages", [])
    if not msgs:
        return

    html_parts = ['<div class="chat-container">']
    for msg in msgs:
        if msg["role"] == "user":
            html_parts.append(
                f'<div class="user-bubble-wrap">'
                f'<div class="user-bubble">{msg["text"]}</div>'
                f'<span class="avatar">🧑‍💻</span>'
                f'</div>'
            )
        else:
            html_parts.append(
                f'<div class="bot-bubble-wrap">'
                f'<span class="avatar">🤖</span>'
                f'<div class="bot-bubble">{msg["text"]}</div>'
                f'</div>'
            )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────
def main():
    setup_page()
    inject_css()

    # ── Session state defaults ──────────────────
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True
    if "last_spoken" not in st.session_state:
        st.session_state.last_spoken = ""

    # ── Route: Login vs Chatbot ─────────────────
    if not st.session_state.logged_in:
        render_login()
        return

    # ── Initialise chat on first load ───────────
    if not st.session_state.messages:
        st.session_state.messages = [_welcome_message()]

    # ── Sidebar ─────────────────────────────────
    render_sidebar()

    # ── Header banner ───────────────────────────
    st.markdown(
        """
        <div class="chat-header">
          <h1>🎓 ICFAI University Hyderabad</h1>
          <p>B.Tech Helpdesk · Powered by AI · IFHE Bot</p>
          <span class="online-badge">Online 24/7</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── TTS toggle ──────────────────────────────
    tts_col1, tts_col2 = st.columns([5, 1])
    with tts_col2:
        st.session_state.tts_enabled = st.checkbox(
            "🔊 Voice", value=st.session_state.tts_enabled, help="Enable/disable voice responses"
        )

    # ── Chat history ────────────────────────────
    render_chat()
    st.markdown("---")

    # ── Input row ───────────────────────────────
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            label="",
            placeholder="💬 Ask me anything about ICFAI B.Tech…",
            key="user_input_field",
            label_visibility="collapsed",
        )
    with col_send:
        send_clicked = st.button("Send 📨", use_container_width=True)

    # ── Voice input widget ──────────────────────
    col_voice, col_info = st.columns([1, 3])
    with col_voice:
        components.html(VOICE_COMPONENT_HTML, height=68)
    with col_info:
        st.markdown(
            "<div class='voice-status'>🎤 Click <b>Speak</b>, ask your question aloud, "
            "then hit <b>Send</b> after the text appears above.</div>",
            unsafe_allow_html=True,
        )

    # ── Handle submission ───────────────────────
    if send_clicked and user_input.strip():
        user_text = user_input.strip()
        bot_response = find_response(user_text)

        st.session_state.messages.append({"role": "user", "text": user_text})
        st.session_state.messages.append({"role": "bot", "text": bot_response})
        st.session_state.last_spoken = bot_response
        st.rerun()

    # ── TTS: speak the last bot response ────────
    if (
        st.session_state.tts_enabled
        and st.session_state.last_spoken
        and len(st.session_state.messages) > 1
    ):
        speak_text_js(st.session_state.last_spoken)
        st.session_state.last_spoken = ""   # clear so it doesn't repeat on next rerun

    # ── Suggested questions (shown only at start) ──
    if len(st.session_state.messages) <= 2:
        st.markdown(
            "<p style='color:rgba(255,255,255,0.6); font-size:0.88rem; margin-bottom:8px;'>"
            "💡 <b>Try asking:</b></p>",
            unsafe_allow_html=True,
        )
        sample_qs = [
            "What is the fee structure?",
            "When are the semester exams?",
            "How do I apply for a hostel room?",
            "Tell me about placement opportunities",
            "What is the minimum attendance required?",
            "How do I get a bonafide certificate?",
            "What branches are available at ICFAI?",
            "How do I login to the ERP portal?",
            "Tell me about ICFAI scholarships",
        ]
        with st.container():
            cols = st.columns(3)
            for i, q in enumerate(sample_qs):
                with cols[i % 3]:
                    st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
                    if st.button(q, key=f"sq_{i}"):
                        bot_resp = find_response(q)
                        st.session_state.messages.append({"role": "user", "text": q})
                        st.session_state.messages.append({"role": "bot", "text": bot_resp})
                        st.session_state.last_spoken = bot_resp
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
