# ringko — 왼손 검지 IMU 두벌식 자음 입력 데이터 수집

Arduino Nano 33 BLE(rev1, LSM9DS1)를 **왼손 검지 중위지골(둘째 마디)** 에 착용하고 노트북 두벌식 키보드를 칠 때의
IMU 신호와 키 입력을 같은 시계로 기록한다. 설계 근거는 `docs/pipeline.docx`, `Papers/`(aLeak, QwertyRing).

## 설정 요약
| 항목 | 값 | 이유 |
|---|---|---|
| 착용 | 왼손 검지 중위지골 | QwertyRing: 근위지골은 2·3행 구분 불가, 피치로 행 구분하려면 중위지골 |
| IMU | 가속도+자이로 238Hz 동기, ±8g, ±2000dps | 200Hz 이상에서 터치 검출 유지(QwertyRing), 타격 포화 방지 |
| 자력계 | 80Hz, 별도 기록 | 노트북 자석 간섭 → 기본 분석은 6축, 자력계는 조건부 |
| 시계 동기화 | 1초마다 핑(P) → 직선 + 구간 보정 모델 | 사용자 동작 없이 ms 이하 정렬. fw-2는 32MHz 크리스털(HFXO)을 켬 |
| 키 입력 | Windows 저수준 키보드 훅(물리 키 VK/스캔코드) | 한글 IME 상태와 무관(Tk 이벤트는 IME에 깨짐) |
| 정지 확인 | 손가락이 F 위에서 0.3초 멈춰야 다음 글자 표시 | Stage 1 body→world 변환의 정지 기준점 확보 |

## 준비 (Windows)
1. Arduino IDE → `firmware/ring_imu/ring_imu.ino` 업로드 (보드: Arduino Mbed OS Nano Boards → Nano 33 BLE). 라이브러리 불필요.
   업로드 후 **시리얼 모니터는 닫기**.
2. `pip install -r requirements.txt`
3. 수집 중 입력기는 **영문**.

## 실행 (`ringko` 폴더에서)
```
python scripts/check_device.py --tap                                     # 장치 점검
python scripts/collect.py --subject P00 --wear 1 --reps 3 --words 15 --out data/pilot   # 파일럿(~5분)
python scripts/collect.py --subject S01 --wear 1                         # 본 수집(~35분)
python scripts/collect.py --subject S01 --wear 1 --phases rest,calib,isolated    # 나눠서
python scripts/report.py data/raw                                        # 품질 보고
python tests/test_ringko.py                                              # 검증(장치 불필요)
```
`--wear`는 링을 새로 낄 때마다 1, 2, 3…으로 올린다(착용 간 일반화 평가). 중단: Esc 두 번.

## 세션 흐름
축 확인 15초 → 정지 5초 → F 보정 10회 → 개별 키(14자음 × reps, 블록 무작위) → 연속 타이핑(단어) → 정지 5초.
- 축 확인: 링 낀 손을 여러 방향으로 돌렸다 멈추기 → `scripts/check_axes.py`로 자이로 축 부호 검증.
  **원시 자이로는 가속도축 기준 오른손 법칙과 부호가 반대**(파일럿 실측, `ringko/frames.py` GYRO_SIGN=(-1,-1,-1)). 분석 시 곱해서 쓴다.
- 개별 키: F에서 출발 → 키 → F 복귀 후 멈춤. 틀리거나 시간초과면 나중에 다시(최대 3회).
- 연속 타이핑: 자음 = 왼손 검지, 모음·SPACE = 오른손. BackSpace 없이 SPACE까지.
- 단어: wordfreq 한국어 상위 3만 단어 중 쌍자음/ㅒ/ㅖ 없는 2~8자음 단어, 자음별 10회 이상 등장하도록 보정.

## 저장 형식 `<out>/<subject>/<시각>_w<wear>_<phases>/`
| 파일 | 열 |
|---|---|
| `imu.csv` | `seq, t_us`(장치, 래핑 해제), `ax ay az`[g], `gx gy gz`[dps] |
| `mag.csv` | `t_us, mx my mz`[uT] |
| `ping.csv` | `send_ns, recv_ns`(호스트 perf_counter), `t_us`(장치) |
| `keys.csv` | `host_ns, event(down/up), key, trial_id`(cue 중이면 >0) |
| `trials.csv` | `trial_id, uid, phase, kind, target, expect, attempt, settle_ns, cue_ns, still_at_cue, first_key_ns, end_ns, typed, status` |
| `plan.csv`, `meta.json` | 제시 플랜, 조건·장치 헤더·결과 |

`ringko.dataset.load_session()`이 핑으로 시계를 맞춰 IMU에 `host_ns`(키 로그와 같은 시계) 열을 붙인다.
`trial_windows()`는 시도별 IMU 구간(첫 키다운 기준 ±0.8초)을 돌려준다.

## 분석 파이프라인 — F 기준 3차원 자음 추론 (학습 없음)
설계·수학·검증 전체: [`docs/consonant3d.md`](../docs/consonant3d.md). 이전의 2D 기하 디코더·분류기(학습형)는 제거했다.

```
IMU ─▶ 타격 검출 features.detect_presses [+ refine_presses]   ← 기존 코드 그대로(수정 없음, 출력 동일 검증)
    ─▶ 자이로 부호 align3d.choose_gyro_sign                     ← 정지 자세 중력 변화 + F 복귀 닫힘(라벨 없음)
    ─▶ track3d.track_session                                    ← 타격 사이 적분, 타격 ZUPT, F→F 닫힘(∝T³) + 키보드 평면
    ─▶ align3d.fit_alignment                                    ← yaw·손잡이 EM, 배율 = 사전 자음 빈도 모멘트(라벨 없음)
    ─▶ consonant3d.Scorer3D                                     ← F 거리 min/max(keymap3d) + 방향 + 이전 타격 대비 이동
    ─▶ decode.Lexicon.rank_words                                ← 기존 사전 자음열 후보·순위
```
| 모듈 | 역할 |
|---|---|
| `track3d.py` | `WordTrack`(타격별 F 기준 3차원 위치 P, 이동 D, σ, 닫힘 오차), `strikes_from_detection/keylog` |
| `keymap3d.py` | `Keyboard3D`: 키 칸 3차원 위치, 자음별 d_k · d_min · d_max, 키 단위 정규화(`table()`) |
| `consonant3d.py` | `Alignment`, `Scorer3D(mode=sector\|box\|shell)`, `chain_marginals`(자음 사후확률) |
| `align3d.py` | `choose_gyro_sign`, `fit_alignment`(라벨 없음), `fit_from_taps`(탭 몇 번), `moment_scale` |
| `infer3d.py` | `infer_session`: 위 전체를 한 번에 |
| `decode.py` | `Lexicon`, `ConsonantLM`(사전 텍스트로 만든 자음 트라이그램), `domain_vocab` |
| `simulate.py` | 검증용 모의: 부착 방향 회전, 다른 IMU(축 규약·배율·바이어스·잡음·119Hz) |

```
python scripts/eval3d.py data/raw/S01/*                       # IMU만(검출+보정), 세션마다 라벨 없이 정렬
python scripts/eval3d.py data/raw/S01/* --strikes keylog      # 오라클 타격
python scripts/eval3d.py data/raw/S01/* --causal 20           # 앞 20단어로만 정렬(실시간 흉내)
python scripts/eval3d.py data/raw/S01/* --perturb mount       # 모의: 센서 부착 방향 변경
python scripts/eval3d.py data/raw/S01/* --mode shell --align none   # 거리만(완전 무보정)
python scripts/check_axes.py <세션>                            # (참고) 자이로 축 부호 수동 검증
```
