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
