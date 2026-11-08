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

## 분석 파이프라인 (`docs/pipeline.docx` ⑧ 대응)
| 단계 | 구현 | 기본값 (S01 결과로 결정) |
|---|---|---|
| 동기화 | `clock.py`, `dataset.load_session` | 켬 |
| Stage 1 body→world, 평균 제거(ZUPT) | `motion.py`, `features.py` | 켬 |
| Stage 1-(5) 칼만 스무딩 | `motion.kalman_trend`, `MotionParams.kalman_rho` | 끔 (ρ≤1e-5 변화 없음, 1e-3 악화) |
| 자력계 조건부 결합 (hard-iron, ΔB 임계, 요 보정) | `magnet.py`, `MotionParams.mag_db_max` | 끔 = 6축 (효과 없음) |
| Stage 2 피크 검출 + aLeak Algorithm 1 | `features.detect_presses(pace=median\|fft\|fft_session)` | median (fft_session과 동급, 단어별 fft는 부정확) |
| 레이아웃 매칭 · 착용별 회전/오프셋 | `decode.Geometry` (`from_isolated`/`from_layout`, `fit` A·b, `adapt`) | A·b 켬. `adapt`(정답 없는 EM)는 실제 시험 세션에서 악화 → 쓰지 말 것 |
| 채널별 정규화 · 증강(학습 fold만) | `preprocess.py`(윈도우), `features.Augment`(특징) | 평가 시 비교만 |
| 행 구분 (A) QwertyRing 피치 | `decode.RowModel` (윈도우 전체 평균 tilt) | 끔 (행 단독 43~61%지만 디코더 이득 없음) |
| 행 구분 (B) aLeak 내적 매칭 | `Lexicon.rank_words(score="both")`, `kbest_sequences`+`rerank_inner` | 단어 켬 κ=20 |
| Stage 4 타이핑 간격 | `decode.TimingModel` | 단어 켬 τ=1 |
| 사전 (자음열, 빈도 순위, 맥락 사전) | `decode.Lexicon`, `ConsonantLM`, `domain_vocab` | 켬 (맥락 사전은 `--domain`) |

```
python scripts/evaluate.py <세션>                       # 자음/단어 Top-1/3/5 + 구성요소별 결과(단일 세션 CV)
python scripts/ablation.py <세션> [--only kalman,mag,stage2,row,timing,inner,wear,aug]   # 켜고/끄기 비교
python scripts/pipeline_test.py --train <학습> --test <시험> [--calib-words 10] [--domain 단어목록.txt]
```
채택 설정의 시험 세션(20261103_181653, 50단어) 결과: 단어 Top-1 2.4만 어휘 16→20%, 4.4천 어휘 16→26%. 자음 Top-1은 47%로 그대로.
모두 S01 한 번 착용한 데이터에서 얻은 값이므로, 착용 보정은 다시 낀 세션(`--wear 2`)으로 검증해야 한다.
