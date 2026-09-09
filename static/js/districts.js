// District lists, alphabetical within each state (Requirement 1).
//
// KNOWN GAP, not yet fixed (flagged to the user, not silently swallowed):
// these are current administrative district names, but orchestrator/pipeline.py
// passes `${district},IN` straight into OpenWeatherMap's live geocoder
// (agents/weather/weather_agent.py) for the Weather Agent. Several of the
// newer, person-named districts below (e.g. "Alluri Sitharama Raju",
// "NTR", "YSR Kadapa", "Sri Sathya Sai") are very unlikely to resolve as
// real place names there. The Price Agent does NOT use district at all
// (agents/price/price_agent.py only filters on State + Commodity), so this
// gap is Weather-only. Needs a real per-district geocoding test before demo day.
export const DISTRICTS = {
  "Andhra Pradesh": [
    "Alluri Sitharama Raju", "Anakapalli", "Anantapur", "Annamayya", "Bapatla",
    "Chittoor", "East Godavari", "Eluru", "Guntur", "Kakinada", "Konaseema",
    "Krishna", "Kurnool", "Nandyal", "NTR", "Palnadu", "Parvathipuram Manyam",
    "Prakasam", "Sri Potti Sriramulu Nellore", "Sri Sathya Sai", "Srikakulam",
    "Tirupati", "Visakhapatnam", "Vizianagaram", "West Godavari", "YSR Kadapa",
  ],
  "Telangana": [
    "Adilabad", "Bhadradri Kothagudem", "Hanamkonda", "Hyderabad", "Jagtial",
    "Jangaon", "Jayashankar Bhupalapally", "Jogulamba Gadwal", "Kamareddy",
    "Karimnagar", "Khammam", "Kumuram Bheem Asifabad", "Mahabubabad",
    "Mahabubnagar", "Mancherial", "Medak", "Medchal-Malkajgiri", "Mulugu",
    "Nagarkurnool", "Nalgonda", "Narayanpet", "Nirmal", "Nizamabad",
    "Peddapalli", "Rajanna Sircilla", "Ranga Reddy", "Sangareddy", "Siddipet",
    "Suryapet", "Vikarabad", "Wanaparthy", "Warangal", "Yadadri Bhuvanagiri",
  ],
};

export const CROPS = ["Tomato", "Potato"];

// UI display only (frontend-layer, not touching agents/disease/disease_agent.py's
// own answer text, which keeps crop names in English by an existing team
// decision). Standard, everyday Telugu words for these two vegetables -- not
// technical/agricultural jargon like the disease names, so not flagged for a
// fluent-speaker review the way TELUGU_DISEASE_NAMES was, but still worth a
// second pair of eyes since this wasn't written by a Telugu speaker.
export const TELUGU_CROP_NAMES = {
  Tomato: "టమాటా",
  Potato: "బంగాళదుంప",
};

export const TELUGU_STATE_NAMES = {
  "Andhra Pradesh": "ఆంధ్రప్రదేశ్",
  "Telangana": "తెలంగాణ",
};

// Native Telugu spellings of the district names above -- these are Telugu
// place names to begin with (this UI's English list is itself the
// transliteration), so this is mostly a script switch, not a translation.
// NOT reviewed by a fluent Telugu speaker (same caveat as TELUGU_CROP_NAMES,
// but more names and several very recently created districts -- e.g. the
// 2022 AP splits like "Alluri Sitharama Raju", "Konaseema", "Palnadu" --
// where I have lower confidence in the exact standard spelling). Displayed
// alongside the English name (not instead of it) specifically so a mistake
// here is visible/checkable rather than silently trusted. Worth a native
// speaker's pass before a real demo.
export const TELUGU_DISTRICT_NAMES = {
  // Andhra Pradesh
  "Alluri Sitharama Raju": "అల్లూరి సీతారామరాజు",
  "Anakapalli": "అనకాపల్లి",
  "Anantapur": "అనంతపురం",
  "Annamayya": "అన్నమయ్య",
  "Bapatla": "బాపట్ల",
  "Chittoor": "చిత్తూరు",
  "East Godavari": "తూర్పు గోదావరి",
  "Eluru": "ఏలూరు",
  "Guntur": "గుంటూరు",
  "Kakinada": "కాకినాడ",
  "Konaseema": "కోనసీమ",
  "Krishna": "కృష్ణా",
  "Kurnool": "కర్నూలు",
  "Nandyal": "నంద్యాల",
  "NTR": "ఎన్టీఆర్",
  "Palnadu": "పల్నాడు",
  "Parvathipuram Manyam": "పార్వతీపురం మన్యం",
  "Prakasam": "ప్రకాశం",
  "Sri Potti Sriramulu Nellore": "శ్రీ పొట్టి శ్రీరాములు నెల్లూరు",
  "Sri Sathya Sai": "శ్రీ సత్యసాయి",
  "Srikakulam": "శ్రీకాకుళం",
  "Tirupati": "తిరుపతి",
  "Visakhapatnam": "విశాఖపట్నం",
  "Vizianagaram": "విజయనగరం",
  "West Godavari": "పశ్చిమ గోదావరి",
  "YSR Kadapa": "వైఎస్సార్ కడప",
  // Telangana
  "Adilabad": "ఆదిలాబాద్",
  "Bhadradri Kothagudem": "భద్రాద్రి కొత్తగూడెం",
  "Hanamkonda": "హనుమకొండ",
  "Hyderabad": "హైదరాబాద్",
  "Jagtial": "జగిత్యాల",
  "Jangaon": "జనగామ",
  "Jayashankar Bhupalapally": "జయశంకర్ భూపాలపల్లి",
  "Jogulamba Gadwal": "జోగులాంబ గద్వాల",
  "Kamareddy": "కామారెడ్డి",
  "Karimnagar": "కరీంనగర్",
  "Khammam": "ఖమ్మం",
  "Kumuram Bheem Asifabad": "కుమురం భీం ఆసిఫాబాద్",
  "Mahabubabad": "మహబూబాబాద్",
  "Mahabubnagar": "మహబూబ్‌నగర్",
  "Mancherial": "మంచిర్యాల",
  "Medak": "మెదక్",
  "Medchal-Malkajgiri": "మేడ్చల్-మల్కాజ్‌గిరి",
  "Mulugu": "ములుగు",
  "Nagarkurnool": "నాగర్‌కర్నూల్",
  "Nalgonda": "నల్గొండ",
  "Narayanpet": "నారాయణపేట",
  "Nirmal": "నిర్మల్",
  "Nizamabad": "నిజామాబాద్",
  "Peddapalli": "పెద్దపల్లి",
  "Rajanna Sircilla": "రాజన్న సిరిసిల్ల",
  "Ranga Reddy": "రంగారెడ్డి",
  "Sangareddy": "సంగారెడ్డి",
  "Siddipet": "సిద్దిపేట",
  "Suryapet": "సూర్యాపేట",
  "Vikarabad": "వికారాబాద్",
  "Wanaparthy": "వనపర్తి",
  "Warangal": "వరంగల్",
  "Yadadri Bhuvanagiri": "యాదాద్రి భువనగిరి",
};
