import json

a = {
    str(t.get("topic_id", t.get("post_id")))
    for t in json.load(open(
        r"Preprocessing\data\processed\weibo_ced\topics_time_sliced.json",
        encoding="utf-8"
    ))
}

g = json.load(open(
    r"Group_Detection_Tracking\data_output\topics_groups.json",
    encoding="utf-8"
))

g = {
    str(k) for k in (
        g.keys() if isinstance(g, dict)
        else [x.get("topic_id") for x in g]
    )
}

print(len(a), len(g), len(a - g), len(g - a))