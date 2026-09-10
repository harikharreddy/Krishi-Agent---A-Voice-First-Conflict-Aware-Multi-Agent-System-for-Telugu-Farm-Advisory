// District lists, alphabetical within each state (Requirement 1).
//
// These are current administrative district names, and orchestrator/pipeline.py
// passes `${district},IN` straight into OpenWeatherMap's live geocoder
// (agents/weather/weather_agent.py) for the Weather Agent. 25 of these 59
// (mostly newer, person-named districts like "Alluri Sitharama Raju", "NTR",
// "YSR Kadapa", plus a few compound/real-town names OpenWeatherMap's static
// city database doesn't have) failed live geocoding when tested on
// 2026-09-09 -- see backend/geocoding_overrides.py, which now maps each of
// those 25 to a verified nearby place name that DOES resolve. All 59
// districts have live-weather coverage as of that fix; the override table
// is what to check/extend first if a district ever starts failing again
// (e.g. a future OpenWeatherMap database change). The Price Agent does NOT
// use district at all (agents/price/price_agent.py only filters on State +
// Commodity), so this was always Weather-only.
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
// decision). Standard, everyday Telugu words for these two vegetables.
// Reviewed Sep 2026 (Claude, checked against standard everyday Telugu
// usage, not a human native-speaker sign-off) -- both correct as-is.
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
// Reviewed Sep 2026 (Claude, checked all 59 -- including the 2022 AP splits
// like "Alluri Sitharama Raju", "Konaseema", "Palnadu" -- against standard
// official Telugu district-name spellings) -- no errors found. This is a
// careful automated check, not a human native speaker's sign-off; displayed
// alongside the English name (not instead of it) specifically so a mistake
// would still be visible/checkable rather than silently trusted. A native
// speaker's pass is still the higher bar before a real demo, but this is no
// longer a known/unchecked gap.
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

// Real market names -> state, built from data/price_history_ap_telangana.csv
// (the actual historical price data agents/price/price_agent.py reads --
// same file, not a separate guess). 157 cleaned/deduplicated entries out of
// 212 raw distinct market strings in that file (dropped messy fragments
// like "Nr. Co-op Central Bank" and joined-address entries). Used only for
// a soft "does this typed mandi look like it's in the other state"
// warning -- NOT exhaustive (a farmer's real nearest mandi may legitimately
// not be in this list at all, in which case no warning shows either way).
// Deliberately NOT attempting a "how far is this mandi from your district"
// check -- would need real distance/coordinate data this project doesn't
// have, and the CSV's districts are the old pre-reorganization names (see
// the DISTRICTS gap note above), which don't map cleanly onto the current
// district list without guessing. State-level is as far as real data
// supports.
export const MANDI_STATE_HINTS = {
  "Adilabad": "TG",
  "Adoni": "AP",
  "Akkaiahpalem": "AP",
  "Allagadda": "AP",
  "Alur": "AP",
  "Alwal": "TG",
  "Amadalavalasa": "AP",
  "Amalapuram": "AP",
  "Amangal": "TG",
  "Anantapur": "AP",
  "Armoor": "TG",
  "Asifabad": "TG",
  "Bangarupalem": "AP",
  "Bapulapadu": "AP",
  "Bhimavaram": "AP",
  "Bhimunipatnam": "AP",
  "Bobbili": "AP",
  "Bowenpally": "TG",
  "CSMLS Point": "AP",
  "Chevella": "TG",
  "Chilakaluripet": "AP",
  "Chinnoar": "TG",
  "Chirala": "AP",
  "Chittoor": "AP",
  "Chittor": "AP",
  "Cuddapah": "AP",
  "Dhone": "AP",
  "Dibbala Bazar": "AP",
  "Eluru": "AP",
  "Eluru-II": "AP",
  "Erragadda": "TG",
  "Falaknama": "TG",
  "Fatehkhanpet": "AP",
  "Gaddiannaram": "TG",
  "Gajapathinagaram": "AP",
  "Gajuwaka": "AP",
  "Gajwel": "TG",
  "Gandhi Irvin Park": "AP",
  "Gopala Patnam Cent.": "AP",
  "Gudimalkapur": "TG",
  "Gudiwada": "AP",
  "Gudur": "AP",
  "Guntakal": "AP",
  "Guntur": "AP",
  "Hanmarkonda": "TG",
  "Hindupur": "AP",
  "Hyderabad": "TG",
  "Ibrahimputnam": "TG",
  "Ichapuram": "AP",
  "Jaggayyapeta": "AP",
  "Jainath": "TG",
  "Kakinada": "AP",
  "Kalikiri": "AP",
  "Kalwakurthy": "TG",
  "Kamalapuram": "AP",
  "Kanchekacherla": "AP",
  "Kandukur": "AP",
  "Karimnagar": "TG",
  "Kavali": "AP",
  "Khammam": "TG",
  "Kodad": "TG",
  "Kothagudem": "TG",
  "Kothapatnam B.Stand": "AP",
  "Kothapeta": "AP",
  "Kothpet": "AP",
  "Kovvur": "AP",
  "Kukatpally": "TG",
  "Kuppam": "AP",
  "Kurnool": "AP",
  "Kurupam": "AP",
  "L B Nagar": "TG",
  "Lawyer pet": "AP",
  "Laxettipet": "TG",
  "Machilipatnam": "AP",
  "Madanapalli": "AP",
  "Madnoor": "TG",
  "Mahabubabad": "TG",
  "Mahabubnagar": "TG",
  "Mahboob Manison": "TG",
  "Mahbubnagar": "TG",
  "Mancharial": "TG",
  "Mangalagiri": "AP",
  "Marapally": "TG",
  "Marripalem": "AP",
  "Medak": "TG",
  "Mehdipatnam": "TG",
  "Mehndipatnam": "TG",
  "Miryalaguda": "TG",
  "Miryalguda": "TG",
  "Mulakalacheruvu": "AP",
  "Mydukur": "AP",
  "Mylavaram": "AP",
  "Nagur": "TG",
  "Nalgonda": "TG",
  "Nandigama": "AP",
  "Nandyal": "AP",
  "Narasaraopet": "AP",
  "Narsapuram": "AP",
  "Nellore": "AP",
  "Nidadavolu": "AP",
  "Nirmal": "TG",
  "Nizamabad": "TG",
  "Nuzvid": "AP",
  "Ongole": "AP",
  "Pakala": "AP",
  "Palakole": "AP",
  "Palamaner": "AP",
  "Pamarru": "AP",
  "Parvathipuram": "AP",
  "Patamata": "AP",
  "Pattikonda": "AP",
  "Pedagantiyada": "AP",
  "Peddapuram": "AP",
  "Peddavaltair": "AP",
  "Piler": "AP",
  "Pochamma Maidan": "TG",
  "Ponnur": "AP",
  "Proddatur": "AP",
  "Punganur": "AP",
  "Qutubullahpur": "TG",
  "Rajahmundry": "AP",
  "Ramachandrapuram": "AP",
  "Ramakrisnapuram": "TG",
  "Ramchandrapuram": "TG",
  "Ravulapalem": "AP",
  "Ravulapelem": "AP",
  "Rayachoti": "AP",
  "Rompicherla": "AP",
  "Sadasivpet": "TG",
  "Saluru": "AP",
  "Sangareddy": "TG",
  "Sardarnagar": "TG",
  "Saroornagar": "TG",
  "Sathupalli": "TG",
  "Sattupalli": "TG",
  "Secunderabad": "TG",
  "Seethammadhara": "AP",
  "Shadnagar": "TG",
  "Shankarapally": "TG",
  "Siddipet": "TG",
  "Somala": "AP",
  "Srikakulam": "AP",
  "Srikalahasti": "AP",
  "Suryapeta": "TG",
  "Tirupati": "AP",
  "Ungatur": "AP",
  "Valmikipuram": "AP",
  "Vanasthalipuram": "TG",
  "Vantamamidi": "TG",
  "Vayalapadu": "AP",
  "Venkateswarnagar": "TG",
  "Vijayawada": "AP",
  "Vikarabad": "TG",
  "Visakhapatnam": "AP",
  "Vuyyur": "AP",
  "Wanaparthy Road": "TG",
  "Warangal": "TG",
};
