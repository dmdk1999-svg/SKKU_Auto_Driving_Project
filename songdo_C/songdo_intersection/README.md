# 송도 C 교차로 학습환경

송도 C 교차로의 관측된 전체 삼거리와 주변 7개 도로 구간을 구성했다. 실제 궤적에서 확인한 세 방향 유입과 6개 직진·좌회전·우회전 경로를 사용한다. 동쪽 진출로는 2차로에서 3차로로 이어진다. SUMO 내부 연결 차로와 신호, 차량 간 추종·양보를 포함한다.

## 실행

ZIP을 풀고 demo.cmd를 두 번 클릭하면 AM3 세션의 SUMO-GUI가 열린다. 전용 Python 환경에 처음 필요한 패키지를 설치할 때 인터넷 연결이 필요하다. 이 PC에서는 설치·실행을 확인했다. Python 3.12 x64를 사용한다.

- 파란 차량은 북동쪽 1_2에서 진입한다.
- 주황 차량은 남서쪽 2_2에서 진입한다.
- 보라 차량은 동쪽 3_3에서 진입한다.
- 재생/일시정지 버튼으로 진행을 제어한다. 다시 보려면 Ctrl+R로 시뮬레이션을 다시 불러오고 재생한다.
- SUMO가 이전에 설정한 중단점에서 멈추면 재생 버튼으로 계속할 수 있다.

수동 실행은 이 폴더에서 다음과 같다.

```powershell
$python = Join-Path $env:TEMP 'songdo-sumo128-py312/Scripts/python.exe'
& $python run.py --session AM3
& $python run.py --session PM5
& $python run.py --session AM3 --audit
& $python verify.py
```

SUMO에서 sessions/AM3/scenario.sumocfg를 직접 열어도 같은 네트워크·교통·신호·0.2초 물리 설정을 불러온다. GUI 최초 확대 범위는 network/view.xml에 있다. 다른 세션은 각 sessions 폴더에서 고를 수 있다.

## 복원한 도로

| 도로 구간 | 방향 | 차로 수 |
|---|---|---:|
| 1_2 | 북동쪽에서 교차로로 진입 | 4 |
| 1_1 | 교차로에서 북동쪽으로 진출 | 3 |
| 2_2 | 남서쪽에서 교차로로 진입 | 3 |
| 2_1 | 교차로에서 남서쪽으로 진출 | 3 |
| 3_3 | 동쪽에서 교차로로 진입 | 3 |
| 3_1 | 교차로에서 동쪽으로 진출 | 2 |
| 3_2 | 동쪽 진출로가 이어지는 확장 구간 | 3 |

차로 경계 폴리곤을 실제 좌표계로 변환하고, 차량 진행 방향으로 각 도로의 방향을 정했다. 로컬 원점을 옮겼으며 원래 도로의 방향을 유지했다. 교차로 내부 연결 곡선은 SUMO netconvert가 생성했다. 도로 경계는 드론 관측 범위다.

등록된 경로는 1_2→2_1, 1_2→3_1→3_2, 2_2→1_1, 2_2→3_1→3_2, 3_3→1_1, 3_3→2_1이다. 드물게 기록된 유턴과 출발·도착 방향을 알 수 없는 궤적은 제외하고 사유별 수를 sessions/manifest.json에 남겼다.

## 데이터가 반영되는 방식

입력은 2022-10-04_C.zip의 AM1~AM5, PM1~PM5와 segmentations.zip의 C.csv다. 10개 세션의 원본 차량 ID 3,660개 중 출발·도착 경로와 연속 추적이 확인된 2,805개를 사용한다. 차량별 시작 시각, 시작 위치, 차체 길이·폭, 기록에서 측정한 95백분위 속도를 이용한다. 짧은 궤적, 불완전 경로, 추적 단절 등은 기록에 남기고 제외한다.

실제 진행 경로에 맞는 차로를 확보하기 위해 진입 도로의 마지막 안정적인 차로 라벨을 출발 차로로 사용하고, 해당 경로로 이어지는 가장 가까운 SUMO 차로에 배치한다. SUMO의 안전 삽입 조건에 따라 실제 투입 시각이 늦어질 수 있다. 초기 속도는 그 위치에서 허용되는 삽입 속도다. 그래서 원자료의 모든 시각·속도·위치가 동일하게 재현되는 구성은 아니다.

NPC는 실행 중 SUMO IDM 추종과 SL2015 차로 변경으로 제어한다. 앞차가 멈추거나 ego가 끼어들면 새로운 상황에 반응한다. 위치를 매 스텝 기록 좌표로 덮어쓰는 코드는 주행 환경에 사용하지 않는다. tau 1.3초, 일반 감속 3.5m/s², 비상 감속 7m/s², 정지 간격 1.2m, 행동 갱신 0.6초는 명시적으로 정한 모델 파라미터다. 이 값들을 송도 개별 운전자마다 학습해 추정하는 작업은 포함하지 않았다.

원본 궤적을 비교하거나 이후 전문가 행동 추출에 사용할 수 있게 sessions/<세션>/expert.csv.gz도 제공한다. 0.2초 시간 격자에서의 차량 중심 x/y, 속도, 경로가 들어 있다. 데이터 레코드의 위치는 세계 좌표이고, 학습 관측은 ego 기준 좌표이므로 이 CSV를 그대로 정책 입력으로 사용하지 않는다.

## 신호 복원 범위

제공된 자료에는 실제 신호 제어기의 녹색·황색·적색 로그가 없다. 따라서 신호는 차량의 접근로 이탈 시각을 근거로 추정했다.

서로 충돌하지 않는 주도로 직진·우회전, 북동쪽 좌회전, 동쪽 진입의 세 보호 현시를 구성했다. 황색 3초와 전적색 2초를 두고, 수요에 비례한 녹색 시간으로 90~180초 주기와 오프셋을 탐색한다. 관측된 통과 시각이 해당 녹색 현시에 들어가는 비율을 이동 그룹별로 평균해 적합도를 계산했다. 세션별 적합도는 약 0.70~0.95다. 실제 현장 신호의 정확한 복원 여부는 확인되지 않았다. 특히 두 짧은 세션의 신호 추정에는 관측 주기 수가 부족할 수 있다.

검증은 설정된 신호에 대한 정지·출발과 충돌 없는 교차 통행을 확인한다. 교통량·대기시간·실제 통과 시각의 현장 재현 정확도까지 검증한 결과는 아니다.

## 에이전트 학습 연결

environment.py의 SongdoIntersectionEnv는 Gymnasium 환경이다. 실제 기록에서 복원한 차량 한 대를 ego로 선택하고 제어권을 에이전트에 넘긴다. 다른 차량은 반응형 NPC로 유지된다. 내부 교차로에서는 정해진 회전 경로를 따라간다.

```python
import numpy as np
from environment import SongdoIntersectionEnv

env = SongdoIntersectionEnv(session="AM3", safe_control=False)
try:
    obs, info = env.reset(seed=42, options={"route": "1_2>3_2"})
    for _ in range(100):
        # 연결할 학습 정책이 이 두 값을 출력하도록 한다.
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

action[0]은 정규화 가감속이다. 양수 최대 2m/s², 음수 최대 7m/s²다. action[1]은 최대 0.8m/s의 횡속도 명령이며 차로 내 이동과 차로 변경을 지원한다. 요청은 SUMO sublane 물리에 반영되고 실제 횡속도는 관측으로 돌아온다. 교차로 내부의 조향은 연결 차로를 따른다.

safe_control=False에서는 ego의 기본 안전 제어를 해제해 접촉과 적색 통과가 발생할 수 있다. 접촉 −100, 적색 통과 −50, 목적지 도착 +20으로 종료한다. 진행거리 m당 +0.05, 스텝당 −0.002를 사용한다. 시간 제한은 기본 1,500스텝(300초)이다. safe_control=True에서는 SUMO 안전 제약도 적용하므로 학습 에이전트가 요청한 행동과 실제 실행값이 달라질 수 있다.

관측은 교차로를 다루기 위한 Dict 공간이다. 이전 단일 직선 구간 정책을 바로 불러오려면 관측 어댑터가 필요하다.

| 관측 | 형태 | 내용 |
|---|---|---|
| ego | 14 | 속도, 가속도, 차로 중심 편위, 횡속도, 경로 잔여거리, 출구 방향 회전각, 다음 신호 상태와 거리, 차로 인덱스, 좌·우 차로 존재, 교차로 내부 여부, 차 길이·폭 |
| neighbors | 8×8 | 60m 내 가까운 8대의 종·횡 상대위치, 종·횡 상대속도, 상대 방향 sin/cos, 길이·폭 |
| neighbor_mask | 8 | 각 이웃 행에 실제 차량이 존재하는지 표시 |

모든 값은 −1~1로 정규화한다. 신호 상태는 적색 −1, 없음 0, 황색 0.5, 녹색 1이다. 여러 작업자를 사용하면 각각 별도 프로세스에서 환경을 만든다. libsumo는 같은 프로세스에서 여러 시뮬레이터 인스턴스를 동시에 실행할 수 없다. Gymnasium 공간은 FlattenObservation 래퍼나 Dict 관측을 지원하는 정책과 연결할 수 있다.

## 검증

10개 세션에서 2,805대 전부 목적 진출로까지 도착했다. 접촉 0건, 텔레포트 0건이다. 6개 이동 경로 모두 연결되었고 도로와 내부 차로 사이 끝점 간격의 최댓값은 0m였다.

AM3 통합 검사에서는 345대가 교차로 내부 차로를 통과했다. 적색에 정지한 차량 76대, 녹색 신호를 관측하면서 다시 출발한 차량 52대를 확인했다. 앞차에 12초 정지 명령을 주자 지정한 후행차가 약 11.2m/s에서 0m/s까지 감속했고, 이후 전체 교통이 다시 빠져나갔다. 이 시험에서도 접촉·텔레포트는 없었다.

차량의 원자료 투입 위치와 추정 신호 상태가 다르면 일부 투입 직후 비상 제동 경고가 발생할 수 있다. 기본 검증 로그에 이 경고와 투입 지연을 남겼다. 시뮬레이션 충돌 이벤트는 별도로 집계했다.

검증 파일은 validation/all_sessions.json, validation/integration.json 및 세션별 trips.xml, statistics.xml, collisions.xml, warnings.log다. validation/sumo_intersection.png는 실제 SUMO 화면이고 network.png는 원본 차로 폴리곤 위에 생성한 연결 차로를 겹친 도면이다.

Gymnasium 공식 환경 검사와 6개 이동 경로별 reset·step을 통과했다. 이번 패키지는 학습환경과 데이터 준비까지 제공한다. 정책 학습 결과나 새로운 모델 가중치는 포함하지 않는다.

## 원자료에서 재생성

```powershell
$python = Join-Path $env:TEMP 'songdo-sumo128-py312/Scripts/python.exe'
& $python build.py --archive "D:/download/2022-10-04_C.zip" --segments "D:/download/segmentations.zip"
& $python inspect_map.py
& $python verify.py
```

## 출처

Robert Fonod, Haechan Cho, Hwasoo Yeo, Nikolas Geroliminis (2025), Songdo Traffic 및 Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery. data의 파생 궤적·도로 폴리곤은 원자료에 대한 좌표 이동, 재표본화, 경로 추출을 거쳤다. 원자료와 파생 데이터에 CC BY 4.0 출처 표시를 유지한다. [데이터](https://zenodo.org/records/13828384) · [논문](https://doi.org/10.1016/j.trc.2025.105205) · [라이선스](https://creativecommons.org/licenses/by/4.0/)

SUMO 도로 연결과 신호 모델의 구현은 공식 문서를 따른다. [도로망 형식](https://sumo.dlr.de/docs/Networks/PlainXML.html) · [신호 제어](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html)

