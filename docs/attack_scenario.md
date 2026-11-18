# 공격 시나리오 — IMU 측채널로 한국어 비밀번호 복원

## 위협 모델
한국 사용자가 비밀번호로 **한국어 단어를 영문 자판 상태로 입력**하는 일이 흔하다(예: `사랑` → `tkfkd`).
왼손 검지에 낀 IMU(반지형)만으로 자음 타격 위치를 복원하면, 사전과 맞춰 그 단어를 좁힐 수 있고,
단어가 정해지면 모음까지 포함한 전체 영문 자판 문자열(= 실제 비밀번호)이 결정된다.
센서가 직접 보는 것은 자음(왼손)뿐이고, 모음은 '복원된 단어'에서 나온다.

> 연구 범위: 수집에 동의한 본인/피험자(S01~S03, NEW…)의 데이터에 대한 복원 가능성 시연.
> 실제 타인의 입력을 몰래 수집하거나, 복원 문자열을 실제 계정에 시도하는 것은 범위 밖이다.

## 공격 파이프라인
```
IMU 반지 ─▶ 녹화(제시 단어·라벨 없음)  record_free.py
         ─▶ 단어 경계·자음 위치·사전 매칭  decode_free.py
         ─▶ 단어별 후보 Top-N(CSV)  →  brute-force(각 단어에 to_keys() 적용해 영문 비번열 생성·대조)
```
영문 변환(`ringko.hangul.to_keys`)과 대조는 brute-force 단계에서 처리하므로 디코더는 **한국어 단어 후보만** 낸다.

## 실행 (ringko 폴더에서)

### 1) 후보 단어 사전 넓히기 (이름 + 사전 밖 단어)
`data/wordlist/custom.txt` 에 공격 대상이 쓸 법한 이름·별명·사전 밖 단어를 한 줄에 하나씩 적는다.

### 2) 녹화 (또는 이미 녹화한 세션 사용)
```
python scripts/record_free.py --subject TARGET --wear 1
```

### 3) 단어 후보 추출 → CSV
```
python scripts/decode_free.py data/free/TARGET/<세션> --k 20 --add-words data/wordlist/custom.txt --csv cand.csv
```
`cand.csv` 열: `id, t_s, n_strikes, skeletons(자음열 후보), words(단어 후보 Top-20), truth(채점용)…`
- `--k 20` : brute-force 씨앗이므로 Top-3보다 넓게.
- `words` 열의 각 단어에 `to_keys()` 를 적용한 문자열이 곧 시도할 비밀번호 후보다.
- 사전에 없는 단어는 `words` 에 안 뜨지만 `skeletons`(자음열)로는 남으므로, 자음열을 씨앗으로 한 2차 생성도 가능.

### 4) brute-force (별도 단계)
`cand.csv` 의 단어 후보 → `to_keys()` → 영문 비밀번호 후보 목록 → 오프라인 해시 대조 등.

## 성능 참고 (자기 데이터, 단어 Top-5 기준)
자음 Top-3 81~95%. 단어 Top-3~5 는 19~62%로 사람·착용에 따라 편차가 크다.
사전 밖 단어는 `custom.txt` 에 넣어야 단어 후보로 복원된다(안 넣으면 자음열로만).
