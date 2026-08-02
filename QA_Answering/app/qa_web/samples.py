"""Demo passages shown in the UI dropdown. Kept out of the blueprint so both the
template and any future seed script can import the same list."""

SAMPLES = [
    {
        "label": "Product manual (2 passages)",
        "question": "How long is the warranty?",
        "contexts": [
            "The XR-500 router supports dual-band Wi-Fi on 2.4 GHz and 5 GHz. Firmware updates are "
            "delivered automatically every Tuesday at 03:00 local time and take about four minutes "
            "to install.",
            "The XR-500 carries a two-year limited warranty covering manufacturing defects. The "
            "warranty is void if the casing is opened. Accessories are covered for ninety days.",
        ],
    },
    {
        "label": "Healthcare guideline",
        "question": "When should blood pressure be re-assessed after a dose change?",
        "contexts": [
            "For adults with uncomplicated hypertension, first-line therapy usually consists of a "
            "thiazide diuretic, an ACE inhibitor, or a calcium channel blocker. Blood pressure "
            "should be re-assessed four weeks after any dose change. Patients with an eGFR below "
            "30 mL/min should be referred to a nephrologist.",
        ],
    },
    {
        "label": "Conflicting sources",
        "question": "Where is the head office?",
        "contexts": [
            "Founded in 2019, the company opened its head office in Bengaluru and a sales office "
            "in Pune.",
            "All corporate functions report to the head office in Bengaluru, which moved to "
            "Whitefield in 2024.",
            "The support centre operates from Hyderabad on weekdays.",
        ],
    },
]
