"""netconvert로 만든 도로망의 비정상 연결과 형상을 보정하는 모듈.

osm_road.write_network()가 도로망을 만든 직후 호출한다. OSM 형상은 정확하지
않으므로 OSM과 비교하지 않고, 만들어진 도로망 자체에서 이상한 곳을 찾아 고친다.

    1) 겹침: 서로 연결되지 않은 도로가 같은 높이에서 겹치면 떼어 놓는다.
         - 같은 도로의 반대 방향 차도끼리 겹치면 양쪽을 절반씩 바깥으로 민다.
         - 같은 방향으로 나란히 겹치면 차로가 적은 쪽을 민다.
         - 그 밖에 (가로지르거나 다른 도로의 반대 방향 차도 위를 지나면)
           차로가 적은 쪽을 올려 입체 교차로 만든다.
    2) 연결: 합류·분기점에서 램프는 진입하는 쪽 바깥 차로에만 잇고, 본선 차로는
       차로 수가 맞으면 하나씩 잇는다. 이어지지 않고 끊긴 차로는 옆 차로로 합류시킨다.
    3) 정렬: 접속부 앞뒤 차로 위치가 맞도록 나가는 edge의 시작점(합류 램프는 끝점)을
       옮기고, 본선 폭 안으로 들어온 램프 구간을 바깥으로 민다.

각 단계는 패치 파일(edge 형상, 연결)을 만들어 netconvert로 기존 망에 덮어쓴다.
"""
import math
import os
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

import sumolib
from sumolib import checkBinary

STEP = 5.0      # 형상 보간 간격 (m)
GAP = 1.0       # 떼어 놓은 두 도로 가장자리 사이 간격 (m)
LIFT = 6.0      # 입체 교차로 올리는 높이 (m)
SLOPE = 0.1     # 들어 올린 높이를 앞뒤로 낮춰 가는 기울기
FLAT = 20.0     # edge 양 끝에서 높이를 바꾸지 않는 거리 (접속부 안쪽) (m)
SPREAD = 0.15   # 떼어 놓은 거리를 앞뒤로 줄여 가는 기울기
TAPER = 40.0    # 끝점 이동을 형상에 나눠 싣는 거리 (m)
PUSH_IN = 30.0  # 램프를 본선 바깥으로 미는 거리를 노드에서부터 채워 가는 거리 (m)


def repair(net_path: str, out_dir: str):
    """도로망을 제자리에서 보정한다."""
    # 접속부 보정으로 램프가 밀려나 새로 겹칠 수 있어 겹침 처리를 다시 한 뒤 한 번 더 정렬한다
    for steps in (("separate", "junctions", "junctions"), ("separate", "junctions")):
        for step in steps:
            if step == "separate":
                _apply(net_path, out_dir, _separate(net_path), {})
            else:
                _apply(net_path, out_dir, *_fix_junctions(net_path))


# ── 도로망 읽기 ─────────────────────────────────────────────────

def _parse(shape: str):
    pts = [[float(v) for v in p.split(",")] for p in shape.split()]
    return [p + [0.0] * (3 - len(p)) for p in pts]


def _load(net_path: str):
    """(sumolib 망, edge 형상, lane 형상). 형상은 높이(z)까지 net.xml에서 직접 읽는다."""
    net = sumolib.net.readNet(net_path, withInternal=False)
    root = ET.parse(net_path).getroot()
    nodes = {j.get("id"): [float(j.get("x")), float(j.get("y")), float(j.get("z", 0))]
             for j in root.findall("junction")}
    edge_shape, lane_shape = {}, {}
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        edge_shape[e.get("id")] = (_parse(e.get("shape")) if e.get("shape")
                                   else [list(nodes[e.get("from")]), list(nodes[e.get("to")])])
        for lane in e.findall("lane"):
            lane_shape[lane.get("id")] = _parse(lane.get("shape"))
    return net, edge_shape, lane_shape


def _apply(net_path: str, out_dir: str, shapes: dict, conns: dict):
    """edge 형상과 연결 변경을 패치 파일로 써서 netconvert로 덮어쓴다.

    conns: {(from edge, to edge): (지울 (fromLane, toLane) 집합, 추가할 집합)}
    """
    if not shapes and not conns:
        return
    edg = os.path.join(out_dir, "repair_patch.edg.xml")
    con = os.path.join(out_dir, "repair_patch.con.xml")
    with open(edg, "w", encoding="utf-8") as f:
        f.write("<edges>\n")
        for eid, pts in shapes.items():
            shape = " ".join(",".join(f"{v:.2f}" for v in p) for p in pts)
            f.write(f'    <edge id="{eid}" shape="{shape}"/>\n')
        f.write("</edges>\n")
    with open(con, "w", encoding="utf-8") as f:
        f.write("<connections>\n")
        for (a, b), (drop, add) in conns.items():
            for fl, tl in sorted(drop):
                f.write(f'    <delete from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n')
            for fl, tl in sorted(add):
                f.write(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n')
        f.write("</connections>\n")
    subprocess.run([checkBinary("netconvert"), "--sumo-net-file", net_path,
                    "--edge-files", edg, "--connection-files", con, "-o", net_path],
                   check=True, capture_output=True)


# ── 기하 도구 ───────────────────────────────────────────────────

def _densify(pts, step: float = STEP):
    """형상 점 사이 간격이 step 이하가 되도록 점을 보간해 넣는다."""
    out = [list(pts[0])]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(1, math.ceil(math.dist(a[:2], b[:2]) / step))
        out += [[a[j] + (b[j] - a[j]) * i / k for j in range(3)] for i in range(1, k + 1)]
    return out


def _cumlen(pts):
    s = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        s.append(s[-1] + math.dist(a[:2], b[:2]))
    return s


def _project(shape, p):
    """polyline 기준 점 p의 위치.

    (오른쪽 +횡거리, 수선의 발 xy, 오른쪽 단위벡터, 진행 단위벡터, 발의 z, 위치)
    위치는 shape 시작 앞이면 "before", 끝 뒤면 "after", 그 사이면 "in".
    """
    best = None
    last = len(shape) - 2
    for k, (a, b) in enumerate(zip(shape[:-1], shape[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = dx * dx + dy * dy
        if seg < 1e-9:
            continue
        t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / seg
        tc = min(1.0, max(0.0, t))
        d = math.hypot(p[0] - a[0] - tc * dx, p[1] - a[1] - tc * dy)
        if best is None or d < best[0]:
            norm = math.sqrt(seg)
            right = (dy / norm, -dx / norm)
            fx, fy = a[0] + t * dx, a[1] + t * dy
            off = (p[0] - fx) * right[0] + (p[1] - fy) * right[1]
            where = "before" if k == 0 and t < 0 else "after" if k == last and t > 1 else "in"
            z = a[2] + tc * (b[2] - a[2])
            best = (d, off, (fx, fy), right, (dx / norm, dy / norm), z, where)
    return best[1:]


def _direction(pts, i):
    a, b = (pts[i], pts[i + 1]) if i + 1 < len(pts) else (pts[i - 1], pts[i])
    d = math.dist(a[:2], b[:2]) or 1.0
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def _width(edge) -> float:
    return sum(lane.getWidth() for lane in edge.getLanes())


def _related(e, o) -> bool:
    """노드를 공유하는 edge (접속부에서 만나므로 겹침 검사 대상이 아님)."""
    return bool({e.getFromNode(), e.getToNode()} & {o.getFromNode(), o.getToNode()})


def _spread(values, ds: float, slope: float):
    """각 점의 값을 앞뒤로 slope 기울기만큼 줄어들며 퍼지게 한다 (최댓값 기준)."""
    out = list(values)
    for i, v in enumerate(values):
        if v <= 0:
            continue
        for j in range(len(values)):
            out[j] = max(out[j], v - abs(i - j) * ds * slope)
    return out


# ── 1) 겹침 ─────────────────────────────────────────────────────

def _separate(net_path: str) -> dict:
    net, shapes, _ = _load(net_path)
    edges = net.getEdges()
    dense = {e.getID(): _densify(shapes[e.getID()]) for e in edges}
    cell = 20.0
    grid = defaultdict(set)
    for e in edges:
        for p in dense[e.getID()]:
            grid[(int(p[0] // cell), int(p[1] // cell))].add(e)

    moves = {e.getID(): [(0.0, (0.0, 0.0))] * len(dense[e.getID()]) for e in edges}
    lifts = {e.getID(): [0.0] * len(dense[e.getID()]) for e in edges}
    crossing = {}

    def crosses(e, o):
        """e가 o의 한쪽에서 반대쪽으로 넘어가면 넘어간 뒤의 쪽(+1 오른쪽, -1 왼쪽), 아니면 0."""
        key = (e.getID(), o.getID())
        if key not in crossing:
            offs = [r[0] for r in (_project(shapes[o.getID()], p) for p in dense[e.getID()])
                    if r[5] == "in"]
            crossing[key] = ((1 if offs[-1] > 0 else -1)
                             if offs and min(offs) < -1.0 and max(offs) > 1.0 else 0)
        return crossing[key]
    for e in edges:
        pts = dense[e.getID()]
        for i, p in enumerate(pts):
            gx, gy = int(p[0] // cell), int(p[1] // cell)
            near = set().union(*(grid[(gx + a, gy + b)] for a in (-1, 0, 1) for b in (-1, 0, 1)))
            for o in near:
                if o is e or _related(e, o):
                    continue
                off, _, right, odir, oz, where = _project(shapes[o.getID()], p)
                need = (_width(e) + _width(o)) / 2 + GAP
                if where != "in" or abs(off) >= need or abs(p[2] - oz) > 3.0:
                    continue
                edir = _direction(pts, i)
                dot = edir[0] * odir[0] + edir[1] * odir[1]
                smaller = (e.getLaneNumber(), e.getID()) < (o.getLaneNumber(), o.getID())
                twin = dot < -0.7 and e.getName() and e.getName() == o.getName()
                # 같은 방향: 나란히 달리거나, 가로질러 넘어간 뒤 옆에 붙어 달리는 구간이면 민다
                beside = dot > 0.7 and crosses(e, o) in (0, 1 if off > 0 else -1)
                if not twin and not beside:             # 입체 교차 → 작은 쪽을 올림
                    if smaller:
                        lifts[e.getID()][i] = max(lifts[e.getID()][i], oz + LIFT - p[2])
                    continue
                if dot > 0 and not smaller:             # 같은 방향: 작은 쪽만 민다
                    continue
                dist = (need - abs(off)) * (0.5 if dot < 0 else 1.0)
                sign = 1.0 if off >= 0 else -1.0        # o에서 멀어지는 쪽
                if dist > moves[e.getID()][i][0]:
                    moves[e.getID()][i] = (dist, (right[0] * sign, right[1] * sign))

    out = {}
    for e in edges:
        eid = e.getID()
        mags = [m for m, _ in moves[eid]]
        if max(mags) <= 0 and max(lifts[eid]) <= 0:
            continue
        pts = [list(p) for p in dense[eid]]
        spread = _spread(mags, STEP, SPREAD)
        dirs = [d for _, d in moves[eid]]
        for i in range(len(pts)):                       # 퍼진 구간은 가장 가까운 원인의 방향을 쓴다
            if spread[i] > 0 and mags[i] <= 0:
                src = min((j for j in range(len(pts)) if mags[j] > 0), key=lambda j: abs(j - i))
                dirs[i] = dirs[src]
            pts[i][0] += dirs[i][0] * spread[i]
            pts[i][1] += dirs[i][1] * spread[i]
        # 끝에서 FLAT m까지는 올리지 않아 접속부의 높이를 앞뒤 도로와 맞춘다
        s = _cumlen(pts)
        for i, h in enumerate(_spread(lifts[eid], STEP, SLOPE)):
            pts[i][2] += min(h, SLOPE * max(0.0, min(s[i], s[-1] - s[i]) - FLAT))
        out[eid] = pts
    return out


# ── 2), 3) 접속부 연결과 정렬 ────────────────────────────────────

def _side(branch_shape, ref_shape, at_end: bool) -> int:
    """램프가 본선의 어느 쪽에 있는지. +1 오른쪽, -1 왼쪽 (노드에서 80m 안에서 가장 먼 점 기준)."""
    pts = _densify(branch_shape)
    if at_end:
        pts = pts[::-1]
    s = _cumlen(pts)
    offs = [_project(ref_shape, p)[0] for p, d in zip(pts, s) if 0 < d <= 80.0] or [0.0]
    far = max(offs, key=abs)
    return 1 if far >= 0 else -1


def _chain(edge, forward: bool, limit: float = 150.0):
    """edge부터 갈림 없이 이어지는 edge들 (limit m까지). forward=False면 거슬러 올라간다."""
    chain, length = [edge], edge.getLength()
    while length < limit:
        nxt = list(chain[-1].getOutgoing()) if forward else list(chain[-1].getIncoming())
        if len(nxt) != 1:
            break
        back = nxt[0].getIncoming() if forward else nxt[0].getOutgoing()
        if len(back) != 1 or nxt[0] in chain:
            break
        chain.append(nxt[0])
        length += nxt[0].getLength()
    return chain if forward else chain[::-1]


def _outer_lanes(n: int, k: int, side: int):
    """n차로 중 side 쪽 바깥 k개 차로 번호 (오른쪽 차로부터)."""
    k = min(k, n)
    return list(range(k)) if side > 0 else list(range(n - k, n))


def _fix_junctions(net_path: str):
    net, shapes, lanes = _load(net_path)
    cur = {}
    for e in net.getEdges():
        for to, cl in e.getOutgoing().items():
            cur[(e.getID(), to.getID())] = {(c.getFromLane().getIndex(), c.getToLane().getIndex())
                                            for c in cl}
    new = dict(cur)
    start_shift, end_shift, pushes = {}, {}, []

    def frame(eid, at_start):
        """edge 끝점(노드 위치)과 그 점에서의 오른쪽 단위벡터."""
        pts = shapes[eid]
        a, b = (pts[0], pts[1]) if at_start else (pts[-2], pts[-1])
        d = math.dist(a[:2], b[:2]) or 1.0
        return (pts[0] if at_start else pts[-1]), ((b[1] - a[1]) / d, -(b[0] - a[0]) / d)

    def lane_at(eid, i, at_start):
        """노드 위치에서 i번 차로 중심 (차로는 edge 형상을 가운데로 펼쳐진다)."""
        p, right = frame(eid, at_start)
        edge = net.getEdge(eid)
        off = ((edge.getLaneNumber() - 1) / 2 - i) * edge.getLane(i).getWidth()
        return (p[0] + right[0] * off, p[1] + right[1] * off)

    def gap(a, b, pairs, extra=(0.0, 0.0)):
        """a 차로 끝 → b 차로 시작의 옆 방향 어긋남 중앙값 (b를 이만큼 옮기면 맞는다).

        노드 위치의 형상 끝점 기준으로 계산한다 (접속부에서 잘린 차로 끝점은 비스듬히
        들어오는 램프에서 옆으로 치우쳐 있다). 옆 방향은 a 끝점 기준이다.
        """
        if not pairs:
            return (0.0, 0.0)
        right = frame(a, at_start=False)[1]
        lat = statistics.median(
            (lane_at(a, f, False)[0] - lane_at(b, t, True)[0] - extra[0]) * right[0]
            + (lane_at(a, f, False)[1] - lane_at(b, t, True)[1] - extra[1]) * right[1]
            for f, t in pairs)
        return (right[0] * lat, right[1] * lat)

    def rank(e):
        return (e.getLaneNumber(), "link" not in e.getType(), e.getID())

    for node in net.getNodes():
        inc, out = node.getIncoming(), node.getOutgoing()
        if node.getType() == "traffic_light" and not (len(inc) == 1 and len(out) == 1):
            continue
        if len(inc) == 2 and len(out) == 1:             # 합류
            o = out[0]
            main, br = sorted(inc, key=rank, reverse=True)
            n, m, k = o.getLaneNumber(), main.getLaneNumber(), br.getLaneNumber()
            side = _side(shapes[br.getID()], shapes[main.getID()], at_end=True)
            outer = _outer_lanes(n, k, side)
            br_pairs = {(i, outer[min(i, len(outer) - 1)]) for i in range(k)}
            if m == n:      # 차로를 같이 쓰는 합류: 램프 차로를 모두 맨 바깥 차로 하나로 (경로 교차 방지)
                br_pairs = {(i, _outer_lanes(n, 1, side)[0]) for i in range(k)}
            if m + k == n:
                shift = k if side > 0 else 0
                main_pairs = {(j, j + shift) for j in range(m)}
            elif m == n:
                main_pairs = {(j, j) for j in range(m)}
            else:
                main_pairs = cur.get((main.getID(), o.getID()), set())
            new[(main.getID(), o.getID())] = main_pairs
            new[(br.getID(), o.getID())] = br_pairs
            d_out = gap(main.getID(), o.getID(), main_pairs)
            start_shift[o.getID()] = d_out
            # 본선과 차로를 같이 쓰는 합류면 램프 끝은 그 차로 바로 바깥에 붙인다
            out_right = frame(o.getID(), at_start=True)[1]
            outside = _width(br) * side if m == n else 0.0
            extra = (d_out[0] + out_right[0] * outside, d_out[1] + out_right[1] * outside)
            d_br = gap(br.getID(), o.getID(), br_pairs, extra=extra)
            end_shift[br.getID()] = (-d_br[0], -d_br[1])
            pushes.append((br, main, side, (_width(main) + _width(br)) / 2, True))
        elif len(inc) == 1 and len(out) == 2:           # 분기
            i_e = inc[0]
            main, br = sorted(out, key=rank, reverse=True)
            n, m, k = i_e.getLaneNumber(), main.getLaneNumber(), br.getLaneNumber()
            side = _side(shapes[br.getID()], shapes[main.getID()], at_end=False)
            outer = _outer_lanes(n, k, side)
            if m + k == n:
                shift = k if side > 0 else 0
                main_pairs = {(j + shift, j) for j in range(m)}
                br_pairs = {(outer[i], i) for i in range(k)}
            elif m == n:
                main_pairs = {(j, j) for j in range(m)}
                br_pairs = {(outer[min(i, len(outer) - 1)], i) for i in range(k)}
            else:
                main_pairs = cur.get((i_e.getID(), main.getID()), set())
                br_pairs = cur.get((i_e.getID(), br.getID()), set())
            new[(i_e.getID(), main.getID())] = main_pairs
            new[(i_e.getID(), br.getID())] = br_pairs
            start_shift[main.getID()] = gap(i_e.getID(), main.getID(), main_pairs)
            # 본선과 차로를 같이 쓰는 분기면 램프 시작은 그 차로 바로 바깥에 붙인다
            in_right = frame(i_e.getID(), at_start=False)[1]
            outside = _width(br) * side if m == n else 0.0
            d = gap(i_e.getID(), br.getID(), br_pairs)
            start_shift[br.getID()] = (d[0] + in_right[0] * outside, d[1] + in_right[1] * outside)
            pushes.append((br, main, side, (_width(main) + _width(br)) / 2, False))
        elif len(inc) == 1 and len(out) == 1:           # 이어지는 도로
            a, b = inc[0], out[0]
            pairs = set(cur.get((a.getID(), b.getID()), set()))
            if not pairs:
                continue
            linked = {f: t for f, t in sorted(pairs)}
            for f in range(a.getLaneNumber()):          # 끊긴 차로는 가장 가까운 차로의 목적지로 합류
                if f not in linked:
                    near = min(linked, key=lambda g: abs(g - f))
                    pairs.add((f, linked[near]))
            new[(a.getID(), b.getID())] = pairs
            one_to_one = {(f, t) for f, t in pairs if sum(1 for g, _ in pairs if g == f) == 1
                          and sum(1 for _, u in pairs if u == t) == 1}
            start_shift[b.getID()] = gap(a.getID(), b.getID(), one_to_one or pairs)

    shapes_out = {}
    for eid in set(start_shift) | set(end_shift):
        ds = start_shift.get(eid, (0.0, 0.0))
        de = end_shift.get(eid, (0.0, 0.0))
        if math.hypot(*ds) < 0.1 and math.hypot(*de) < 0.1:
            continue
        pts = _densify(shapes[eid])
        s = _cumlen(pts)
        length = s[-1] or 1.0
        both = math.hypot(*ds) >= 0.1 and math.hypot(*de) >= 0.1
        taper = min(TAPER, length / 2 if both else length)
        for p, d in zip(pts, s):
            ws = max(0.0, 1.0 - d / taper)
            we = max(0.0, 1.0 - (length - d) / taper)
            p[0] += ds[0] * ws + de[0] * we
            p[1] += ds[1] * ws + de[1] * we
        shapes_out[eid] = pts

    def current(eid):
        return shapes_out.get(eid) or _densify(shapes[eid])

    for br, ref, side, min_off, at_end in pushes:     # 본선 폭 안의 램프 구간을 바깥으로
        # 분기 직후 edge가 짧으면 이어지는 edge까지 한 줄로 이어서 민다
        br_chain = _chain(br, forward=not at_end)
        ref_chain = _chain(ref, forward=not at_end)
        pts, owner = [], []
        for e in br_chain:
            seg = current(e.getID())
            pts += seg
            owner += [(e.getID(), i) for i in range(len(seg))]
        ref_shape = sum((current(e.getID()) for e in ref_chain), [])
        if at_end:
            pts, owner = pts[::-1], owner[::-1]         # 노드 쪽 끝이 앞에 오도록
        node_pt = pts[0]
        end_off = _project(ref_shape, node_pt)[0] * side
        beyond = "before" if at_end else "after"
        moved = set()
        for i in range(1, len(pts)):
            off, foot, right, _, _, where = _project(ref_shape, pts[i])
            dist = math.dist(pts[i][:2], node_pt[:2])
            target = min_off - (min_off - end_off) * max(0.0, 1.0 - dist / PUSH_IN)
            if where == beyond or off * side >= target:
                break
            pts[i][0] = foot[0] + right[0] * side * target
            pts[i][1] = foot[1] + right[1] * side * target
            moved.add(owner[i][0])
        for e in br_chain:
            if e.getID() in moved:
                shapes_out[e.getID()] = [p for p, (eid, _) in sorted(
                    ((p, o) for p, o in zip(pts, owner) if o[0] == e.getID()), key=lambda t: t[1][1])]

    conns = {}
    for key, pairs in new.items():
        old = cur.get(key, set())
        if pairs != old:
            conns[key] = (old - pairs, pairs - old)
    return shapes_out, conns
