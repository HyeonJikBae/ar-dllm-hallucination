"""Explicit, reviewable broad groups for sport / religion / occupation.
label 1 iff gold and answer share at least one broad group; 2 iff both are mapped and share none; '?' if either is unmapped.
Matching: longest phrase first, matched spans are consumed, so 'director of photography' is not also 'director'."""
import re

BALL = "ball_games"
SPORT = {  # phrase -> groups
    **{k: {BALL} for k in ["association football", "football", "soccer", "american football", "rugby union", "rugby league", "rugby", "australian rules football",
        "gaelic football", "baseball", "softball", "cricket", "field hockey", "lacrosse", "basketball", "beach volleyball",
        "volleyball", "handball", "tennis", "table tennis", "golf", "snooker", "sepak takraw", "footballer"]},
    **{k: {"puck_stick"} for k in ["roller hockey", "hockey"]}, "badminton": {"racket"}, "ultimate frisbee": {"disc"},
    "water polo": {BALL, "aquatic"}, "polo": {BALL, "equestrian"}, "ice hockey": {"puck_stick", "winter"},
    **{k: {"winter"} for k in ["alpine skiing", "cross-country skiing", "ski jumping", "ski mountaineering", "nordic combined", "skiing", "snowboarding",
        "figure skating", "speed skating", "bobsleigh", "curling"]},
    **{k: {"aquatic"} for k in ["paralympic swimming", "swimming", "diving", "rowing", "sailing", "dragon boat racing"]},
    "triathlon": {"aquatic", "athletics"},
    **{k: {"combat"} for k in ["brazilian jiu-jitsu", "mixed martial arts", "mixed martial artist", "professional wrestling", "wrestling", "boxing", "judo", "karate", "taekwondo", "sumo", "fencing"]},
    **{k: {"athletics"} for k in ["track and field", "athletics", "artistic gymnastics", "rhythmic gymnastics", "gymnastics", "weightlifting", "rock climbing", "climbing", "gripping"]},
    **{k: {"cycling"} for k in ["road bicycle racing", "track cycling", "cycling", "bmx"]},
    **{k: {"motor"} for k in ["motorcycle speedway", "auto racing"]},
    **{k: {"equestrian"} for k in ["equestrian", "horse riding"]},
    "chess": {"mind"}, "paralympic archery": {"precision"}, "archery": {"precision"},
    **{k: {"video_game"} for k in ["the legend of zelda: ocarina of time", "super mario bros.", "world of warcraft", "super smash bros.", "tetris", "pong"]},
    "dancing": {"dance"}, "kabaddi": {"kabaddi"}, "paralympic": {"paralympic"},
}
REL = {
    **{k: {"christian"} for k in ["roman catholicism", "catholic church", "roman catholic", "catholicism", "catholic", "anglican communion", "anglicanism", "anglican",
        "orthodox christianity", "eastern orthodox church", "greek orthodox church of antioch", "serbian orthodox church", "romanian orthodox church", "orthodox",
        "christianity", "christian", "lutheran church--missouri synod", "lutheran church–missouri synod", "evangelical lutheran church in america", "lutheranism", "lutheran",
        "baptists", "baptist", "american baptist churches usa", "southern baptist convention", "methodism", "methodist", "united methodist church", "free methodist church",
        "christian methodist episcopal church", "presbyterian church (usa)", "presbyterian", "church of scotland", "protestantism", "protestant", "seventh-day adventist",
        "mormonism", "mormon", "quaker", "unitarian universalism", "unitarian universalist", "unitarianism", "unitarian", "mennonite brethren church", "evangelicalism",
        "international pentecostal holiness church", "congregational church", "episcopal church", "christadelphians", "celtic christianity", "prussian union of churches", "non-denominational"]},
    **{k: {"islam"} for k in ["sunni islam", "islam", "muslim"]},
    **{k: {"judaism"} for k in ["judaism", "jewish"]},
    **{k: {"indian_religions"} for k in ["zen buddhism", "buddhism", "hinduism", "jainism", "sikhism"]},
    **{k: {"east_asian"} for k in ["taoism", "confucianism", "shinto"]},
    **{k: {"irreligion"} for k in ["atheism", "agnosticism", "secular humanism"]},
    "anglo-saxon paganism": {"paganism"}, "traditional african religion": {"folk"}, "aymara": {"folk"}, "thelema": {"occult"}, "zoroastrianism": {"zoroastrianism"},
    "rastafarianism": {"rastafari"}, "deism": {"deism"}, "freemasonry": {"freemasonry"},
}
POL, SCI, BUS, REL_, MIL, SPORTS, MED = "politics_law", "science_academia", "business", "religion", "military_security", "sports", "medicine"
MUS, SCR, WRT, VIS, PERF = "music", "screen_stage", "writing_media", "visual_arts", "performing"
OCC = {
    **{k: {MUS} for k in ["composer", "songwriter", "musician", "disc jockey"]},
    "singer": {MUS, PERF},
    **{k: {SCR, PERF} for k in ["actress", "actor", "seiyū", "comedian", "dancer", "presenter"]},
    **{k: {SCR} for k in ["executive producer", "film director", "director of photography", "film industry", "japanese animation", "film", "director"]},
    **{k: {SCR, WRT} for k in ["documentary filmmaker", "screenwriter", "playwright"]},
    **{k: {WRT} for k in ["writer", "author", "novelist", "poet", "journalist", "editor", "publisher", "critic", "printer"]},
    **{k: {VIS} for k in ["photographer", "painter", "artist", "architect", "fashion designer", "fashion"]},
    "model": {PERF, VIS}, "art historian": {VIS, SCI},
    "art collector": {VIS, "business"}, "art dealer": {VIS, "business"},
    **{k: {POL} for k in ["president of the united states", "president of the french republic", "president of poland", "prime minister", "chief minister", "minister of foreign affairs",
        "chief justice", "magistrate", "diplomat", "lawyer", "attorney", "politician", "lord mayor", "mayor", "head of the department", "king of poland", "duke of suffolk",
        "lord of the admiralty", "lord of the manor", "prince", "political science", "director of the office of management and budget", "director of the office of international affairs"]},
    **{k: {SCI} for k in ["professor", "assistant professor", "teacher", "astronomer", "geologist", "biochemist", "biologist", "mathematician", "meteorologist", "economist",
        "statistics", "bioethics", "astrologer", "physicist", "historian", "philosopher", "vice president of the chinese academy of sciences"]},
    **{k: {MED} for k in ["physician", "doctor"]},
    **{k: {BUS} for k in ["chief executive officer", "president and ceo", "businessperson", "businessman", "entrepreneur", "financier", "industrialist", "boston merchant", "executive vice president", "executive director"]},
    **{k: {REL_} for k in ["chief rabbi of israel", "chief rabbi", "rabbi", "priest", "monk", "bishop", "vestal virgin", "orthodox priest"]},
    **{k: {MIL} for k in ["military officer", "british army officer", "british army", "united states army", "soldier", "general", "test pilot", "french resistance", "chief inspector"]},
    **{k: {SPORTS} for k in ["horse trainer", "jockey", "footballer", "cricketer", "hockey player", "professional cyclist", "chess grandmaster", "athlete", "golfer", "wrestler", "poker player"]},
    "chef": {"culinary"},
}
TABLES = {"sport": SPORT, "religion": REL, "occupation": OCC}


def match(rel, text):
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    keys, groups = [], set()
    for k in sorted(TABLES[rel], key=len, reverse=True):
        pat = r"(?<![\w-])" + re.escape(k.strip()) + r"(?![\w-])"
        if re.search(pat, t):
            keys.append(k.strip()); groups |= TABLES[rel][k]; t = re.sub(pat, " ", t)
    return keys, groups


def label(rel, gold, ans):
    gk, gg = match(rel, gold); ak, ag = match(rel, ans)
    if not gg or not ag:
        return "?", "lo", f"unmapped: gold={gk or gold!r} ans={ak or ans[:60]!r}"
    if set(gk) == set(ak):
        return "C", "lo", f"same item {gk}"
    shared = gg & ag
    return ("1", "hi", f"shared group {sorted(shared)}: {gk} / {ak}") if shared else ("2", "hi", f"{sorted(gg)} {gk} vs {sorted(ag)} {ak}")
