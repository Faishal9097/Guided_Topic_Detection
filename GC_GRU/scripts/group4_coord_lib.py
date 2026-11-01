"""Bot-like / coordination signals from Weibo interactions. No labels are used anywhere.

Per user (cumulative interactions up to slice t):
  rapid : share of the user's gaps between own consecutive posts that are <= FAST_GAP_S   (needs >= 3 posts)
  copy  : share of the user's content messages that are COORDINATED COPIES, i.e. the same normalised text is
          posted by >= DUP_MIN_USERS distinct users AND >= 2 other messages of that text fall within +-SYNC_WIN_S
  score = max(rapid, copy);   bot_like = score >= BOT_THR
These are "bot-like" flags, not confirmed bots (Weibo CED has no bot ground truth, no commenter profiles).
Organic copy-paste memes that spread in a short burst will also be flagged: say so in the paper.

Text normalisation: everything from '//@' on is dropped (plain reposts become empty), '回复@x:' prefix, @mentions,
[emoticons], punctuation and the default repost text are removed.  Texts shorter than MIN_TEXT_LEN are 'no content'
(so empty reposts can never count as duplicates; the v2 label rubric made exactly that mistake).
"""
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

FAST_GAP_S = 30
SYNC_WIN_S = 60
SYNC_BIN_S = 60
DUP_MIN_USERS = 3
MIN_TEXT_LEN = 4
BOT_THR = 0.5

COORD_NAMES = ["bot_user_frac", "user_score_mean", "grp_bot_mean", "grp_bot_max", "mixed_grp_frac",
               "bh_edge_frac", "bh_edge_lift", "grp_sync_mean", "grp_dup_mean"]
VOL_NAMES = ["log_interactions", "log_users"]

_REPLY = re.compile(r"回复@[^:：]*[:：]")
_MENTION = re.compile(r"@[^\s:：,，。.!！?？]+")
_EMO = re.compile(r"\[[^\]]{1,8}\]")
_NOISE = re.compile(r"[\W_]+", re.UNICODE)


def norm_text(t):
    t = (t or "").split("//@")[0]
    t = _REPLY.sub("", t)
    t = _MENTION.sub("", t)
    t = _EMO.sub("", t)
    t = t.replace("转发微博", "")
    return _NOISE.sub("", t).lower()


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def prep(interactions, sl=0):
    """interaction dicts (uid, mid, parent, date, text) -> list of light records. uid/mid/parent become str."""
    return [{"uid": str(i["uid"]), "mid": str(i["mid"]), "parent": str(i["parent"]) if i.get("parent") not in (None, "") else "",
             "ts": parse_ts(i["date"]), "nt": norm_text(i.get("text")), "sl": sl} for i in interactions]


def analyse(recs):
    """-> users {uid: dict(n, rapid, copy, score, bot)}, dup (bool per rec), by_user {uid: [rec idx]}"""
    by_user = defaultdict(list)
    for i, r in enumerate(recs):
        by_user[r["uid"]].append(i)
    clus = defaultdict(list)
    for i, r in enumerate(recs):
        if len(r["nt"]) >= MIN_TEXT_LEN:
            clus[r["nt"]].append(i)
    dup = np.zeros(len(recs), bool)
    coord = np.zeros(len(recs), bool)
    for idx in clus.values():
        if len({recs[i]["uid"] for i in idx}) < DUP_MIN_USERS:
            continue
        dup[idx] = True
        ts = sorted(recs[i]["ts"] for i in idx)
        for i in idx:
            t = recs[i]["ts"]
            if bisect_right(ts, t + SYNC_WIN_S) - bisect_left(ts, t - SYNC_WIN_S) - 1 >= 2:
                coord[i] = True
    users = {}
    for u, idx in by_user.items():
        n = len(idx)
        rapid = 0.0
        if n >= 3:
            ts = np.sort([recs[i]["ts"] for i in idx])
            rapid = float((np.diff(ts) <= FAST_GAP_S).mean())
        content = [i for i in idx if len(recs[i]["nt"]) >= MIN_TEXT_LEN]
        copy = float(coord[content].mean()) if content else 0.0
        score = max(rapid, copy)
        users[u] = {"n": n, "rapid": rapid, "copy": copy, "score": score, "bot": score >= BOT_THR}
    return users, dup, by_user


def slice_features(recs, groups):
    """recs: cumulative records up to this slice. groups: {gid: [member uid, ...]}.
    Returns (coord[9], volume[2]) as float32. Everything is a fraction/ratio except the volume control."""
    c = np.zeros(len(COORD_NAMES), np.float32)
    v = np.zeros(len(VOL_NAMES), np.float32)
    if not recs:
        return c, v
    users, dup, by_user = analyse(recs)
    bot = {u: d["bot"] for u, d in users.items()}
    v[:] = [np.log1p(len(recs)), np.log1p(len(users))]
    c[0] = np.mean(list(bot.values()))
    c[1] = np.mean([d["score"] for d in users.values()])

    gb, mixed, sync, gdup = [], 0, [], []
    for mem in groups.values():
        mem = [str(m) for m in mem if str(m) in users]
        if not mem:
            continue
        b = float(np.mean([bot[m] for m in mem]))
        gb.append(b)
        mixed += int(0 < b < 1)
        idx = [i for m in mem for i in by_user[m]]
        if len(idx) >= 5:                                           # share of the group's posts in its busiest minute
            cnt = Counter(int(recs[i]["ts"] // SYNC_BIN_S) for i in idx)
            sync.append(max(cnt.values()) / len(idx))
        cont = [i for i in idx if len(recs[i]["nt"]) >= MIN_TEXT_LEN]
        if cont:
            gdup.append(float(dup[cont].mean()))
    if gb:
        c[2], c[3], c[4] = np.mean(gb), np.max(gb), mixed / len(gb)
    c[7] = np.mean(sync) if sync else 0.0
    c[8] = np.mean(gdup) if gdup else 0.0

    mid2uid = {r["mid"]: r["uid"] for r in recs}                    # only interactions seen so far (no future parents)
    e_tot = e_mix = 0
    nodes = set()
    for r in recs:
        p = r["parent"]
        if p and p in mid2uid and mid2uid[p] != r["uid"]:
            a, b2 = r["uid"], mid2uid[p]
            e_tot += 1
            e_mix += int(bot[a] != bot[b2])
            nodes.update((a, b2))
    if e_tot:
        frac = e_mix / e_tot
        p = float(np.mean([bot[n] for n in nodes]))
        exp = 2 * p * (1 - p)                                       # mixed-edge share if bot flags were random
        c[5] = frac
        c[6] = min(frac / exp, 5.0) if exp > 0 else 0.0
    return c, v
