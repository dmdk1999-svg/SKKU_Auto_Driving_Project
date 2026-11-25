"""OSM 실측 지도로 입체 도로망(한남대교 · 경부고속도로 · 올림픽대로)을 만드는 모듈.

road_builder.build()가 ROAD["osm"]["enabled"]일 때 호출한다.

처리 순서:
    1) env/osm/ 의 원본 OSM에서 쓸 도로만 고르고 구역 밖을 잘라낸다
    2) 끊긴 램프(진출입 상대가 잘려나간 램프)를 제거하고, 다른 도로와 만나지 않는
       말단부는 마지막 접속점에서 end_keep m만 남긴다
    3) 차로 수와 포켓 차로(램프 앞 감속 차로, 좌회전 포켓)를 정한다
    4) 모든 OSM 노드에 높이(ele)를 넣는다
         - low_ways(올림픽대로): z = 0
         - 나머지 본선: z = deck_height
         - 램프: 양 끝 높이 사이를 거리 비례로 이어 경사를 만든다
         - 다른 도로를 노드 공유 없이 가로지르는 구간: 차로가 적은 쪽을 더 올린다
    5) netconvert로 highway.net.xml 컴파일 (진출 램프 앞에는 감속 차로 추가)
    6) net_repair로 비정상 연결과 형상을 보정한다
    7) 설정의 경로(way id 쌍)를 실제 edge 나열로 풀어 route/flow 블록을 만든다

SUMO의 차량 상호작용은 차선/교차로 단위라서, 노드를 공유하지 않는 도로는
xy가 겹쳐도 충돌하거나 앞차로 잡히지 않는다. 입체 교차는 이렇게 성립한다.
"""
import math
import os
import subprocess
import xml.etree.ElementTree as ET

import sumolib
from sumolib import checkBinary

from env import net_repair

ENV_DIR = os.path.dirname(os.path.abspath(__file__))


def _local_xy(lat: float, lon: float, origin) -> tuple:
    """위경도 → 원점 기준 평면 좌표(m). 구역 필터링 전용."""
    lat0, lon0 = origin
    return ((lon - lon0) * 111320.0 * math.cos(math.radians(lat0)),
            (lat - lat0) * 110540.0)


def _in_box(p, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _longest_inside(refs, xy, box):
    """way의 노드 중 box 안에 연속으로 있는 가장 긴 구간."""
    best, cur = [], []
    for r in refs:
        if _in_box(xy[r], box):
            cur.append(r)
        else:
            best, cur = (cur if len(cur) > len(best) else best), []
    return cur if len(cur) > len(best) else best


def _way_tags(way) -> dict:
    return {t.get("k"): t.get("v") for t in way.findall("tag")}


def _set_tag(elem, key: str, value: str):
    for t in elem.findall("tag"):
        if t.get("k") == key:
            t.set("v", value)
            return
    ET.SubElement(elem, "tag", {"k": key, "v": value})


def _set_refs(way, refs):
    for nd in way.findall("nd"):
        way.remove(nd)
    for i, r in enumerate(refs):
        way.insert(i, ET.Element("nd", {"ref": r}))


def _way_refs(way):
    return [nd.get("ref") for nd in way.findall("nd")]


def _select_ways(root, xy, cfg):
    """쓸 way만 남긴다: 본선은 도로별 구역으로 자르고, 램프는 나들목 안의 것만."""
    for way in list(root.findall("way")):
        tags = _way_tags(way)
        refs = _way_refs(way)
        if way.get("id") in cfg.get("drop_ways", ()):
            keep = False
        elif "link" in tags.get("highway", ""):
            keep = all(_in_box(xy[r], cfg["ramp_box"]) for r in refs)
        elif tags.get("name") in cfg["main_roads"]:
            refs = _longest_inside(refs, xy, cfg["main_roads"][tags["name"]])
            keep = len(refs) >= 2
        else:
            keep = False
        if keep:
            _set_refs(way, refs)
        else:
            root.remove(way)


def _drop_signals(root, node_ids):
    """지정한 노드의 신호·횡단보도 태그를 지운다 (netconvert가 그 자리에 신호를 만들지 않는다)."""
    for node in root.findall("node"):
        if node.get("id") in node_ids:
            for tag in node.findall("tag"):
                if tag.get("k") in ("highway", "crossing", "crossing:markings", "traffic_signals"):
                    node.remove(tag)


def _reroute_way_ends(root, spec):
    """way 끝을 다른 노드로 옮긴다. {way id: (남길 마지막 노드, 새 끝 노드)}

    남길 노드 뒤는 지우고 새 끝 노드로 곧게 잇는다 (합류 지점을 옮길 때 쓴다).
    """
    for way in root.findall("way"):
        if way.get("id") in spec:
            keep, end = spec[way.get("id")]
            refs = _way_refs(way)
            _set_refs(way, refs[:refs.index(keep) + 1] + [end])


def _replace_two_way(root, xy, spec):
    """양방향 way를 지우고, 거기서 이어지는 일방통행 way 쌍을 그 길이만큼 직선으로 늘린다.

    spec: {양방향 way id: (정방향 way, 역방향 way)}. 정방향 way는 양방향 way 끝 노드에서
    시작하고, 역방향 way는 그 노드에서 끝나야 한다. 양방향 way와 일방통행 way 쌍이 만나는
    노드는 netconvert가 접속부로 남기고, 그 노드로 모이는 일방통행 way의 첫·끝 구간이
    옆으로 꺾여 있다. 그 노드를 빼고 두 way의 바로 앞 구간 방향으로 늘린다.
    """
    ways = {w.get("id"): w for w in root.findall("way")}
    for wid, (fwd, bwd) in spec.items():
        refs = _way_refs(ways[wid])
        f_refs, b_refs = _way_refs(ways[fwd]), _way_refs(ways[bwd])
        assert f_refs[0] == refs[-1] and b_refs[-1] == refs[-1], \
            f"way {wid}의 끝 노드에서 {fwd}, {bwd}가 이어지지 않습니다."
        length = sum(math.dist(xy[a], xy[b]) for a, b in zip(refs, refs[1:]))
        # 정방향: 둘째 노드에서 셋째 → 둘째 방향으로, 역방향: 끝에서 둘째 노드에서 그 앞 → 그 노드 방향으로
        for way, anchor, ahead, keep in ((ways[fwd], f_refs[1], f_refs[2], f_refs[1:]),
                                         (ways[bwd], b_refs[-2], b_refs[-3], b_refs[:-1])):
            end = _new_node(root, xy, anchor, ahead, -length / math.dist(xy[anchor], xy[ahead]))
            _set_refs(way, [end] + keep if way is ways[fwd] else keep + [end])
        root.remove(ways[wid])


def _prune_dangling_ramps(root):
    """다른 도로와 이어지지 않는 램프를 반복 제거한다.

    구역 밖 도로로 나가는 램프는 잘려서 막다른 길이 되는데, 그대로 두면
    화면에 끊긴 도로로 보이기만 하고 어떤 경로도 지나지 못한다.
    """
    while True:
        ways = root.findall("way")
        starts, ends = {}, {}
        for w in ways:
            for i, r in enumerate(_way_refs(w)):
                # 본선 중간 노드도 램프의 진출입 지점이 될 수 있다
                if i > 0:
                    ends[r] = ends.get(r, 0) + 1
                if i < len(_way_refs(w)) - 1:
                    starts[r] = starts.get(r, 0) + 1
        removed = False
        for w in ways:
            if "link" not in _way_tags(w).get("highway", ""):
                continue
            refs = _way_refs(w)
            # 나가는 곳 없음 또는 들어오는 곳 없음 = 끊긴 램프
            if not starts.get(refs[-1]) or not ends.get(refs[0]):
                root.remove(w)
                removed = True
        if not removed:
            return


def _new_node(root, xy, a: str, b: str, t: float) -> str:
    """노드 a→b 사이 t 비율 위치에 새 노드를 만들고 id를 반환한다."""
    nodes = {n.get("id"): n for n in root.findall("node")}
    new = str(max(int(i) for i in nodes) + 1)
    lerp = lambda key: float(nodes[a].get(key)) + t * (float(nodes[b].get(key)) - float(nodes[a].get(key)))
    # 노드는 기존 노드 뒤, 첫 way 앞에 둔다 (맨 앞에 두면 netconvert가 기존 노드를 읽지 못한다)
    first_way = list(root).index(root.find("way"))
    root.insert(first_way, ET.Element("node", {"id": new, "lat": f"{lerp('lat'):.7f}",
                                               "lon": f"{lerp('lon'):.7f}"}))
    xy[new] = (xy[a][0] + t * (xy[b][0] - xy[a][0]), xy[a][1] + t * (xy[b][1] - xy[a][1]))
    return new


def _trim_ends(root, xy, keep: float, protected):
    """다른 도로와 만나지 않는 말단부를 마지막 접속점에서 keep m만 남기고 자른다.

    막다른 끝 노드에서 갈림 없이 이어지는 노드를 따라가 처음 만나는 접속점(이웃 노드
    3개 이상)까지를 말단부로 본다. protected way(경로의 출발·도착 way)는 한 구간 이상 남긴다.
    같은 도로의 상하행 끝이 나란히 있으면 둘 중 덜 잘리는 쪽에 맞춰 끝 위치를 맞춘다.
    """
    ways = root.findall("way")
    adj, owner = {}, {}
    name = {w.get("id"): _way_tags(w).get("name") for w in ways}
    for w in ways:
        refs = _way_refs(w)
        for a, b in zip(refs, refs[1:]):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        for r in refs:
            owner.setdefault(r, set()).add(w.get("id"))
    tails = []                  # (끝 노드, 접속점 → 끝 노드 경로, 누적 거리, 자를 거리)
    for end in [n for n, s in adj.items() if len(s) == 1]:
        path, prev, cur = [end], None, end
        while True:
            nxt = [m for m in adj[cur] if m != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
            if len(adj[cur]) != 2:
                break
        if len(adj[cur]) < 3:       # 접속점 없이 다른 끝으로 이어진 도로
            continue
        path.reverse()              # 접속점 → 끝
        dist = [0.0]
        for a, b in zip(path, path[1:]):
            dist.append(dist[-1] + math.dist(xy[a], xy[b]))
        cut = keep
        for wid in protected:
            on = sorted(d for n, d in zip(path, dist) if wid in owner[n])
            if on:                  # 그 way의 첫 노드에서 20m는 남긴다
                cut = max(cut, on[0] + 20.0)
        tails.append([end, path, dist, cut])

    # 상하행 끝 맞추기: 같은 이름, 끝 노드가 80m 안이면 잘라내는 길이를 작은 쪽에 맞춘다
    road = lambda n: {name[w] for w in owner[n]}
    for i, a in enumerate(tails):
        for b in tails[i + 1:]:
            if (road(a[0]) & road(b[0])) - {None} and math.dist(xy[a[0]], xy[b[0]]) < 80.0:
                removed = min(a[2][-1] - a[3], b[2][-1] - b[3])
                a[3], b[3] = a[2][-1] - removed, b[2][-1] - removed

    drop, cuts = set(), {}      # cuts: (남는 노드, 지우는 노드) → 그 사이에 넣은 새 끝 노드
    for _, path, dist, cut in tails:
        k = next((i for i, d in enumerate(dist) if d > cut), None)
        if k is None:
            continue
        drop.update(path[k:])
        if cut - dist[k - 1] > 0.5:
            a, b = path[k - 1], path[k]
            cuts[(a, b)] = _new_node(root, xy, a, b, (cut - dist[k - 1]) / (dist[k] - dist[k - 1]))
    for w in ways:
        refs = _way_refs(w)
        out = []
        for i, r in enumerate(refs):
            if r not in drop:
                out.append(r)
                continue
            for n in (refs[i - 1] if i > 0 else None, refs[i + 1] if i + 1 < len(refs) else None):
                if (n, r) in cuts:
                    out.append(cuts[(n, r)])
        if len(out) >= 2:
            _set_refs(w, out)
        else:
            root.remove(w)


def _apply_road_attributes(root, xy, cfg) -> dict:
    """OSM 태그 보정: 제한속도가 없는 도로의 기본값, 차로 수 상한, way별 차로 수·진행 방향, 포켓 차로.

    ({포켓 way id: (차로가 생기는 쪽, 목적지)}, {나뉘어 새로 생긴 way id: 원래 way id})를 반환한다.
    """
    for way in root.findall("way"):
        tags = _way_tags(way)
        if "maxspeed" not in tags:
            kmh = cfg["default_maxspeed_kmh"].get(tags.get("highway"))
            if kmh:
                _set_tag(way, "maxspeed", str(kmh))
        cap = cfg.get("max_lanes", {}).get(tags.get("name"))
        if cap and int(tags.get("lanes", 0)) > cap:
            _set_tag(way, "lanes", str(cap))
        lanes = cfg.get("way_lanes", {}).get(way.get("id"))
        if isinstance(lanes, tuple):
            _set_tag(way, "lanes", str(sum(lanes)))
            _set_tag(way, "lanes:forward", str(lanes[0]))
            _set_tag(way, "lanes:backward", str(lanes[1]))
        elif lanes:
            _set_tag(way, "lanes", str(lanes))
        # 차로별 진행 방향은 turn_lanes로 지정한 way만 쓴다 (OSM 원본 표기는 버린다)
        for tag in way.findall("tag"):
            if tag.get("k") == "turn:lanes":
                way.remove(tag)
    origin = _apply_partial_lanes(root, cfg.get("way_lanes_until", {}))
    pockets = _apply_pockets(root, xy, cfg.get("pocket_lanes", {}), origin)
    _apply_turn_lanes(root, cfg.get("turn_lanes", {}), origin)
    return pockets, origin


def _apply_turn_lanes(root, spec, origin):
    """way 끝 교차로의 차로별 진행 방향(turn:lanes)을 지정한다. {way id: turn:lanes 값}

    way가 앞에서 여러 조각으로 나뉘었으면 마지막 조각에만 붙인다.
    """
    for wid, turns in spec.items():
        pieces = [w for w in root.findall("way") if origin.get(w.get("id"), w.get("id")) == wid]
        starts = {_way_refs(w)[0] for w in pieces}
        last = next(w for w in pieces if _way_refs(w)[-1] not in starts)
        _set_tag(last, "turn:lanes", turns)


def _apply_partial_lanes(root, spec) -> dict:
    """way의 시작부터 지정 노드까지만 차로 수를 바꾼다. {way id: (차로 수, 노드 id)}

    way를 그 노드에서 둘로 나눠 앞부분을 새 way(id = 기존 최대 id + 1)로 만든다.
    {새 way id: 원래 way id}를 반환한다.
    """
    next_id = max(int(w.get("id")) for w in root.findall("way")) + 1
    origin = {}
    for way in list(root.findall("way")):
        if way.get("id") not in spec:
            continue
        lanes, node = spec[way.get("id")]
        refs = _way_refs(way)
        cut = refs.index(node)
        head = ET.fromstring(ET.tostring(way))
        head.set("id", str(next_id))
        origin[str(next_id)] = way.get("id")
        next_id += 1
        _set_refs(head, refs[:cut + 1])
        _set_tag(head, "lanes", str(lanes))
        _set_refs(way, refs[cut:])
        root.append(head)
    return origin


def _apply_pockets(root, xy, spec, origin) -> dict:
    """노드 앞 일정 길이에 차로를 하나 더한다 (램프 진입 전 감속 차로, 좌회전 포켓).

    spec: {way id: [(노드 id, 길이 m, "right" | "left"[, 포켓 차로만 가는 way id]), ...]}
    way를 노드 앞 길이 지점과 노드에서 나눠 가운데 조각(새 way)의 차로를 하나 늘린다.
    {포켓 way id: (차로가 생기는 쪽, 목적지 way id 또는 None)}을 반환한다
    (net_repair가 차로 연결에 쓴다).
    """
    next_id = max(int(w.get("id")) for w in root.findall("way")) + 1
    sides = {}
    for wid, items in spec.items():
        for node, length, side, *target in items:
            way = next(w for w in root.findall("way")
                       if wid in (w.get("id"), origin.get(w.get("id"))) and node in _way_refs(w)[1:])
            refs = _way_refs(way)
            i = refs.index(node)
            k, d = i, 0.0
            while k > 0 and d + math.dist(xy[refs[k - 1]], xy[refs[k]]) < length:
                d += math.dist(xy[refs[k - 1]], xy[refs[k]])
                k -= 1
            if k == 0:
                head, start = [], refs[0]
            else:
                seg = math.dist(xy[refs[k - 1]], xy[refs[k]])
                start = _new_node(root, xy, refs[k], refs[k - 1], (length - d) / seg)
                head = refs[:k] + [start]
            pieces = [(head, None), ([start] + refs[k:i + 1], "pocket"), (refs[i:], None)]
            lanes = int(_way_tags(way).get("lanes", "1").split(";")[0])
            first = True
            for part, kind in pieces:
                if len(part) < 2:
                    continue
                elem = way if first else ET.fromstring(ET.tostring(way))
                if not first:
                    elem.set("id", str(next_id))
                    origin[str(next_id)] = origin.get(wid, wid)
                    next_id += 1
                    root.append(elem)
                _set_refs(elem, part)
                if kind:
                    _set_tag(elem, "lanes", str(lanes + 1))
                    sides[elem.get("id")] = (side, target[0] if target else None)
                first = False
    return sides


def _segments_cross(a, b, c, d) -> bool:
    """평면에서 선분 ab와 cd가 서로 교차하는지."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (side(a, b, c) * side(a, b, d) < 0
            and side(c, d, a) * side(c, d, b) < 0)


def _assign_elevation(root, xy, cfg):
    """노드 높이 계산. 본선은 고정, 램프 노드는 이웃과의 거리 가중 평균으로 푼다.

    (조화 보간: 램프 어디서든 양 끝 높이 사이를 거리 비례로 잇는 경사가 된다)
    """
    height = float(cfg["deck_height"])
    z, neighbors = {}, {}
    low_segments, ramp_segments, deck_segments = [], [], []
    for way in root.findall("way"):
        tags = _way_tags(way)
        refs = _way_refs(way)
        segments = list(zip(refs[:-1], refs[1:]))
        is_ramp = "link" in tags.get("highway", "")
        if tags.get("name") not in cfg["low_roads"]:
            lanes = tags.get("lanes", "").split(";")[0]
            rank = (int(lanes) if lanes.isdigit() else 1 if is_ramp else 2, not is_ramp)
            deck_segments += [(rank, a, b) for a, b in segments]
        if not is_ramp:
            is_low = tags.get("name") in cfg["low_roads"]
            level = 0.0 if is_low else height
            for r in refs:
                # 본선끼리 만나는 노드는 낮은 쪽 우선 (올림픽대로 높이 유지)
                z[r] = min(z.get(r, level), level)
            if is_low:
                low_segments += segments
        else:
            ramp_segments += segments
        for a, b in segments:
            w = 1.0 / max(math.dist(xy[a], xy[b]), 1.0)
            neighbors.setdefault(a, []).append((b, w))
            neighbors.setdefault(b, []).append((a, w))

    # 올림픽대로를 가로지르는 램프 구간은 고가이므로 그 위 높이로 고정한다
    for ra, rb in ramp_segments:
        if any(_segments_cross(xy[ra], xy[rb], xy[la], xy[lb])
               for la, lb in low_segments if not {ra, rb} & {la, lb}):
            z.setdefault(ra, float(cfg["ramp_clearance"]))
            z.setdefault(rb, float(cfg["ramp_clearance"]))

    # 본선 높이의 도로끼리 노드를 공유하지 않고 평면에서 가로지르면 입체 교차다.
    # 차로가 적은 쪽(같으면 램프)의 그 구간 양 끝 노드를 overpass_clearance만큼 올린다.
    over = height + float(cfg["overpass_clearance"])
    boxes = [(min(xy[a][0], xy[b][0]), max(xy[a][0], xy[b][0]),
              min(xy[a][1], xy[b][1]), max(xy[a][1], xy[b][1])) for _, a, b in deck_segments]
    raised = set()
    for i, (r1, a, b) in enumerate(deck_segments):
        for j in range(i + 1, len(deck_segments)):
            r2, c, d = deck_segments[j]
            bi, bj = boxes[i], boxes[j]
            if bi[1] < bj[0] or bj[1] < bi[0] or bi[3] < bj[2] or bj[3] < bi[2]:
                continue
            if {a, b} & {c, d} or not _segments_cross(xy[a], xy[b], xy[c], xy[d]):
                continue
            raised |= {a, b} if r1 < r2 else {c, d}
    for n in raised:
        if z.get(n, height) >= height:      # 올림픽대로 높이나 그 위 램프 구간은 그대로 둔다
            z[n] = over

    free = [n for n in neighbors if n not in z]
    for n in free:
        z[n] = height
    for _ in range(2000):
        for n in free:
            total = sum(w for _, w in neighbors[n])
            z[n] = sum(z[m] * w for m, w in neighbors[n]) / total

    for node in root.findall("node"):
        if node.get("id") in z:
            _set_tag(node, "ele", f"{z[node.get('id')]:.2f}")


def write_network(out_dir: str, cfg: dict) -> str:
    """OSM을 가공해 highway.net.xml을 만들고 경로를 반환한다."""
    tree = ET.parse(os.path.join(ENV_DIR, "osm", cfg["source"]))
    root = tree.getroot()
    xy = {n.get("id"): _local_xy(float(n.get("lat")), float(n.get("lon")), cfg["origin"])
          for n in root.findall("node")}

    _select_ways(root, xy, cfg)
    _replace_two_way(root, xy, cfg.get("two_way_replace", {}))
    _reroute_way_ends(root, cfg.get("reroute_way_end", {}))
    _drop_signals(root, cfg.get("drop_signals", ()))
    _prune_dangling_ramps(root)
    _trim_ends(root, xy, float(cfg["end_keep"]),
               {w for pair in cfg["routes"].values() for w in pair})
    pockets, origin = _apply_road_attributes(root, xy, cfg)
    pieces = lambda wid: {w.get("id") for w in root.findall("way")
                          if origin.get(w.get("id"), w.get("id")) == wid}
    drops = [(pieces(a), pieces(b)) for a, b in cfg.get("drop_connections", [])]
    _assign_elevation(root, xy, cfg)

    osm_path = os.path.join(out_dir, "highway.osm.xml")
    tree.write(osm_path, encoding="utf-8", xml_declaration=True)

    net_path = os.path.join(out_dir, "highway.net.xml")
    args = [checkBinary("netconvert"),
            "--osm-files", osm_path,
            "--osm.elevation", "true",        # 위에서 넣은 ele 태그를 z로 사용
            "--geometry.remove", "true",
            "--junctions.join", "true",       # OSM의 잘게 나뉜 교차로 노드를 합침
            "--no-turnarounds", "true",
            "--osm.turn-lanes", "true",       # turn:lanes 태그대로 차로별 진행 방향을 정한다
            "--tls.guess-signals", "true",
            "--tls.join", "true",
            "--keep-edges.by-vclass", "passenger",
            "--remove-edges.isolated", "true",
            "--output.street-names", "true",  # 도로 이름 (net_repair가 양방향 차도를 구분할 때 사용)
            "-o", net_path]
    subprocess.run(args + ["--ramps.guess", "false"], check=True, capture_output=True)

    # 진출 램프 앞에는 감속 차로를 추가한다. 합류 램프는 accel_lane_roads 본선이면
    # 가속 차로를 두고, 아니면 차로를 늘리지 않고 본선 맨 오른쪽 차로로 합류시킨다.
    on_ramps = _on_ramp_edges(net_path)
    accel = [e for e, name in on_ramps if name in cfg["accel_lane_roads"]]
    plain = [e for e, name in on_ramps if name not in cfg["accel_lane_roads"]]
    subprocess.run(args + ["--ramps.guess", "true",
                           "--ramps.set", ",".join(accel),
                           "--ramps.unset", ",".join(plain)],
                   check=True, capture_output=True)
    net_repair.repair(net_path, out_dir, pockets, drops)
    _set_split_signals(net_path, out_dir, cfg.get("split_signal", {}))
    return net_path


def _set_split_signals(net_path: str, out_dir: str, spec: dict):
    """신호 교차로를 방향별 단독 신호로 바꾼다. {진입 way id: 녹색 시간(초)}

    적은 순서대로 진입로마다 [녹색, 황색 YELLOW초, 전체 적색 ALL_RED초] 현시를 둔다.
    진입로 앞 노드가 같은 신호에 묶여 있으면 그 연결은 이어지는 진입로와 같이 켠다.
    """
    if not spec:
        return
    YELLOW, ALL_RED = 4, 1
    root = ET.parse(net_path).getroot()
    base = lambda eid: eid.lstrip("-").partition("#")[0]
    links = {}                                  # 신호 id → {연결 번호: 진입 way id}
    conns = [c for c in root.findall("connection") if c.get("tl")]
    for c in conns:
        way = base(c.get("from"))
        if way not in spec:
            way = base(c.get("to"))             # 진입로 앞 노드의 연결
        if way in spec:
            links.setdefault(c.get("tl"), {})[int(c.get("linkIndex"))] = way
    path = os.path.join(out_dir, "signal_patch.tll.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<tlLogics>\n")
        for tl in root.findall("tlLogic"):
            if tl.get("id") not in links:
                continue
            n = len(tl.find("phase").get("state"))
            ways = links[tl.get("id")]
            assert len(ways) == n, f"신호 {tl.get('id')}의 연결 중 진입 way를 모르는 것이 있습니다."
            f.write(f'    <tlLogic id="{tl.get("id")}" type="static" '
                    f'programID="{tl.get("programID")}" offset="{tl.get("offset")}">\n')
            for way, green in spec.items():
                on = {i for i, w in ways.items() if w == way}
                if not on:
                    continue
                for duration, light in ((green, "G"), (YELLOW, "y"), (ALL_RED, "r")):
                    state = "".join(light if i in on else "r" for i in range(n))
                    f.write(f'        <phase duration="{duration}" state="{state}"/>\n')
            f.write("    </tlLogic>\n")
        f.write("</tlLogics>\n")
    subprocess.run([checkBinary("netconvert"), "--sumo-net-file", net_path,
                    "--tllogic-files", path, "-o", net_path], check=True, capture_output=True)


def _on_ramp_edges(net_path: str) -> list:
    """램프가 합류한 직후의 본선 (edge id, 도로 이름) 목록 (들어오는 edge 2개 중 하나가 램프인 노드의 나가는 edge)."""
    net = sumolib.net.readNet(net_path, withInternal=False)
    edges = []
    for node in net.getNodes():
        inc, out = node.getIncoming(), node.getOutgoing()
        if len(inc) == 2 and len(out) == 1 and any("link" in e.getType() for e in inc):
            edges.append((out[0].getID(), out[0].getName()))
    return edges


def _way_edges(net, way_id: str):
    """way에서 만들어진 edge들을 진행 순서로 반환 (#0, #0-AddedOffRampEdge, #1-AddedOnRampEdge, #1 ...).

    netconvert가 덧붙이는 감속 구간(OffRamp)은 원래 edge 뒤에, 가속 구간(OnRamp)은 앞에 온다.
    """
    def order(e):
        num, _, added = e.getID().partition("#")[2].partition("-")
        return (int(num) if num else 0, {"AddedOnRampEdge": -1, "AddedOffRampEdge": 1}.get(added, 0))
    edges = [e for e in net.getEdges()
             if e.getID().partition("#")[0] == way_id]
    if not edges:
        raise ValueError(f"way {way_id}가 도로망에 없습니다. 구역(main_roads/ramp_box)을 확인하세요.")
    return sorted(edges, key=order)


def resolve_routes(net_path: str, cfg: dict) -> dict:
    """{경로 이름: [edge id...]}. way의 맨 앞 edge에서 맨 끝 edge까지 최단 경로.

    경로가 (출발, 경유, 도착) way 세 개면 경유 way의 맨 앞 edge를 지나는 최단 경로다.
    """
    net = sumolib.net.readNet(net_path, withInternal=False)
    routes = {}
    for name, ways in cfg["routes"].items():
        stops = ([_way_edges(net, ways[0])[0]]
                 + [_way_edges(net, w)[0] for w in ways[1:-1]]
                 + [_way_edges(net, ways[-1])[-1]])
        path = [stops[0]]
        for a, b in zip(stops, stops[1:]):
            leg, _ = net.getShortestPath(a, b)
            if not leg:
                raise ValueError(f"경로 {name}: way {' → '.join(ways)} 로 이어지는 길이 없습니다.")
            path += leg[1:]
        routes[name] = [e.getID() for e in path]
    return routes


def demand_block(net_path: str, cfg: dict, flow_type: str, traffic: dict) -> str:
    """route/flow 블록. 도착 간격은 지수분포라 에피소드마다 교통 패턴이 달라진다."""
    routes = resolve_routes(net_path, cfg)
    lines = [f'    <route id="{name}" edges="{" ".join(edges)}"/>'
             for name, edges in routes.items()]
    depart_lane = cfg.get("flow_depart_lane", {})
    for name, vph in cfg["flows"].items():
        lines.append(f'    <flow id="flow_{name}" type="{flow_type}" route="{name}" '
                     f'begin="0" end="100000" period="exp({float(vph) / 3600.0:.5f})" '
                     f'departLane="{depart_lane.get(name, "best")}" '
                     f'departSpeed="{traffic["depart_speed"]}"/>')
    return "\n".join(lines)
