# ringkey — 데이터 수집

왼손 검지 IMU 링(Arduino Nano 33 BLE) + 노트북 키보드 로그를 한 시계로 기록하는 수집 도구.
PROTOCOL.md 설계(14자음 × 50, F 보정 10, 연속 타이핑 150단어, 착용/세션/회전 조건)를 그대로 구현.

## 준비
1. `firmware/ring_logger/ring_logger.ino`를 보드에 업로드 (라이브러리: Arduino_LSM9DS1)
2. `pip install -r requirements.txt` (tkinter는 파이썬 배포본에 포함; Linux는 `sudo apt install python3-tk`)
3. 수집 중에는 **입력기를 영문(QWERTY)** 으로. (한글 상태면 경고가 뜨고 meta.json에 `ime_suspected`가 기록됨)

## 순서
```bash
# 0) 파일럿(본 수집 전 필수): 장치 점검 → 짧은 런으로 전체 흐름/동기화 확인. 결과는 data/pilot에 따로 저장
python scripts/check_device.py --seconds 10 --tap     # 실제 레이트/손실/노이즈/탭 스파이크 확인
python scripts/collect.py --subject P00 --session s1 --wear wear0 --site middle_phalanx --blocks 3 --words 15 --out data/pilot
# 본 수집
python scripts/check_device.py --seconds 10 --tap     # 파일럿: 실제 레이트/손실/노이즈/탭 스파이크 확인
python scripts/make_prompts.py --subject S01 --session s1 --wear wear0 --out p.csv   # (선택) 프롬프트 미리보기
python scripts/collect.py --subject S01 --session s1 --wear wear0 --rot 0            # 본 수집(약 53분)
python scripts/collect.py --subject S01 --session s1 --wear wear1 --rot 0            # 링 빼고 재착용
python scripts/audit_data.py data/raw                 # PROTOCOL 대비 진행 현황/경고
```
긴 세션은 `--only isolated` / `--only continuous` 로 나눠 수집(런마다 sync가 자동으로 앞뒤에 붙음).
회전 조건은 `--rot 90` 등. 중단은 Esc 두 번(해당 런은 불완전으로 표시되어 audit에서 제외).

## 한 런의 진행
sync(스페이스 5회) → F 보정 10회 → 개별 키(14자음 × 50, 블록 무작위) → 연속 타이핑 → sync(5회)

## 저장 형식 `data/raw/<subject>/<session>/<wear>_rot<deg>/run_<시각>_<only>/`
| 파일 | 내용 |
|---|---|
| `imu.csv` | `host_ns, seq, t_us_raw, t_us(래핑 해제), ax..az[g], gx..gz[dps], mx..mz[uT], mag_new` |
| `keys.csv` | `host_ns, event(down/up), key, trial_id(큐 중이면 >0)` |
| `trials.csv` | 시도별 `cue_ns, first_key_ns, end_ns, typed, status, gyro_peak_dps` (오입력/시간초과 포함) |
| `prompts.csv` | 제시한 전체 플랜 |
| `sync.json` | 시계 오프셋 모델 (`sync.dev_to_host_s`로 장치 시각 → 키보드 시계) |
| `meta.json` | 피험자/세션/착용/회전, 완료 여부, IMU 레이트·손실, sync 품질 |

시계: `host_ns`, 키, trial 시각은 모두 같은 `perf_counter_ns`. IMU 샘플의 정확한 시각은 `t_us`(장치)에
`sync.json`의 오프셋을 적용해서 얻는다(`host_ns`는 USB 도착 시각이라 지터가 있음).

## 알아둘 점
- 자력계는 20Hz로 갱신되고 나머지 줄은 직전 값 반복(`mag_new=0`). 기본 분석은 6축.
- 수집 단어는 wordfreq 상위 단어에서 자음 키 2~8개, 쌍자음/ㅒ/ㅖ 없는 것만 사용. 실제 비밀번호/개인정보 금지.
- 검증: `python tests/test_pipeline.py` (OS 무관, 가짜 시리얼 사용)
- Windows: 포트는 `COM3` 형태(자동 탐색 실패 시 `--port COM3`). Arduino IDE 시리얼 모니터가 열려 있으면 포트를 못 엽니다.
- `requirements.txt`는 ASCII만 유지(Windows pip가 cp949로 읽음).
