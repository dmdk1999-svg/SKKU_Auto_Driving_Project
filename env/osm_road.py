"""OSM 실측 지도로 입체 도로망(한남대교 · 경부고속도로 · 올림픽대로)을 만드는 모듈.

road_builder.build()가 ROAD["osm"]["enabled"]일 때 호출한다.

처리 순서:
    1) env/osm/ 의 원본 OSM에서 쓸 도로만 고르고 구역 밖을 잘라낸다
    2) 끊긴 램프(진출입 상대가 잘려나간 램프)를 제거한다
    3) 모든 OSM 노드에 높이(ele)를 넣는다
         - low_ways(올림픽대로): z = 0
         - 나머지 본선: z = deck_height
         - 램프: 양 끝 높이 사이를 거리 비례로 이어 경사를 만든다
         - 다른 도로를 노드 공유 없이 가로지르는 구간: 차로가 적은 쪽을 더 올린다
    4) netconvert로 highway.net.xml 컴파일 (진출 램프 앞에는 감속 차로 추가)
    5) net_repair로 비정상 연결과 형상을 보정한다
    6) 설정의 경로(way id 쌍)를 실제 edge 나열로 풀어 route/flow 블록을 만든다

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


def _apply_road_attributes(root, cfg):
    """OSM 태그 보정: 제한속도가 없는 도로의 기본값, 차로 수 상한, way별 차로 수."""
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
        if lanes:
            _set_tag(way, "lanes", str(lanes))
    _apply_partial_lanes(root, cfg.get("way_lanes_until", {}))


def _apply_partial_lanes(root, spec):
    """way의 시작부터 지정 노드까지만 차로 수를 바꾼다. {way id: (차로 수, 노드 id)}

    way를 그 노드에서 둘로 나눠 앞부분을 새 way(id = 기존 최대 id + 1)로 만든다.
    """
    next_id = max(int(w.get("id")) for w in root.findall("way")) + 1
    for way in list(root.findall("way")):
        if way.get("id") not in spec:
            continue
        lanes, node = spec[way.get("id")]
        refs = _way_refs(way)
        cut = refs.index(node)
        head = ET.fromstring(ET.tostring(way))
        head.set("id", str(next_id))
        next_id += 1
        _set_refs(head, refs[:cut + 1])
        _set_tag(head, "lanes", str(lanes))
        _set_refs(way, refs[cut:])
        root.append(head)


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
    _prune_dangling_ramps(root)
    _apply_road_attributes(root, cfg)
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
            "--tls.guess-signals", "true",
            "--tls.join", "true",
            "--keep-edges.by-vclass", "passenger",
            "--remove-edges.isolated", "true",
            "--output.street-names", "true",  # 도로 이름 (net_repair가 양방향 차도를 구분할 때 사용)
            "-o", net_path]
    subprocess.run(args + ["--ramps.guess", "false"], check=True, capture_output=True)

    # 진출 램프 앞에만 감속 차로를 추가한다. 합류 램프는 차로를 늘리지 않고
    # 본선 맨 오른쪽 차로로 합류시키므로 ramp guessing에서 제외한다.
    on_ramps = _on_ramp_edges(net_path)
    subprocess.run(args + ["--ramps.guess", "true",
                           "--ramps.unset", ",".join(on_ramps)],
                   check=True, capture_output=True)
    net_repair.repair(net_path, out_dir)
    return net_path


def _on_ramp_edges(net_path: str) -> list:
    """램프가 합류한 직후의 본선 edge id 목록 (들어오는 edge 2개 중 하나가 램프인 노드의 나가는 edge)."""
    net = sumolib.net.readNet(net_path, withInternal=False)
    edges = []
    for node in net.getNodes():
        inc, out = node.getIncoming(), node.getOutgoing()
        if len(inc) == 2 and len(out) == 1 and any("link" in e.getType() for e in inc):
            edges.append(out[0].getID())
    return edges


def _way_edges(net, way_id: str):
    """way에서 만들어진 edge들을 진행 순서로 반환 (#0, #0-AddedOffRampEdge, #1 ...)."""
    def order(e):
        num, _, added = e.getID().partition("#")[2].partition("-")
        return (int(num) if num else 0, 1 if added else 0)
    edges = [e for e in net.getEdges()
             if e.getID().partition("#")[0] == way_id]
    if not edges:
        raise ValueError(f"way {way_id}가 도로망에 없습니다. 구역(main_roads/ramp_box)을 확인하세요.")
    return sorted(edges, key=order)


def resolve_routes(net_path: str, cfg: dict) -> dict:
    """{경로 이름: [edge id...]}. way의 맨 앞 edge에서 맨 끝 edge까지 최단 경로."""
    net = sumolib.net.readNet(net_path, withInternal=False)
    routes = {}
    for name, (src, dst) in cfg["routes"].items():
        path, _ = net.getShortestPath(_way_edges(net, src)[0], _way_edges(net, dst)[-1])
        if not path:
            raise ValueError(f"경로 {name}: way {src} → {dst} 로 이어지는 길이 없습니다.")
        routes[name] = [e.getID() for e in path]
    return routes


def demand_block(net_path: str, cfg: dict, flow_type: str, traffic: dict) -> str:
    """route/flow 블록. 도착 간격은 지수분포라 에피소드마다 교통 패턴이 달라진다."""
    routes = resolve_routes(net_path, cfg)
    lines = [f'    <route id="{name}" edges="{" ".join(edges)}"/>'
             for name, edges in routes.items()]
    for name, vph in cfg["flows"].items():
        lines.append(f'    <flow id="flow_{name}" type="{flow_type}" route="{name}" '
                     f'begin="0" end="100000" period="exp({float(vph) / 3600.0:.5f})" '
                     f'departLane="best" departSpeed="{traffic["depart_speed"]}"/>')
    return "\n".join(lines)
