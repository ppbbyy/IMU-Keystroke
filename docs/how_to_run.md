# 새 사용자 단어 입력 → 후보 확인 (명령어 모음)

새로운 사람이 라벨 없이 단어를 치고, 그 단어가 후보 안에 있는지 확인하기까지의 전체 명령어.
모두 **`ringko` 폴더 안에서** (cmd 기준) 실행한다. 학습·재수집·라벨 입력은 필요 없다.

## 0. (처음 한 번만) 준비
필요한 패키지 설치. 한 번만 하면 다음부터는 생략.
```bash
pip install -r requirements.txt
```

## 1. 폴더 이동
받은 위치에 맞게 경로를 바꾼다.
```bash
cd C:\Users\samsung-user\Downloads\IMU-Keystroke-3Dver\ringko
```

## 2. 녹화하기
새 사용자가 링을 끼고 단어를 친다. `NEW3`는 사람마다 다른 이름으로.
```bash
python scripts/record_free.py --subject NEW3 --wear 1
```
- 창이 뜨면 단어를 친다. **자음 = 왼손 검지, 모음·스페이스 = 오른손.**
- 단어마다: F 위에서 잠깐 멈춤 → 단어 입력 → F로 돌아와 0.5초 이상 멈춤.
- 입력기는 **영문** 상태로.
- 끝내려면 **Esc 두 번.**
- 끝나면 저장 폴더 이름이 화면에 나온다. 예: `data/free/NEW3/20261118_xxxxxx_w1_free`

## 3. 단어가 후보에 있는지 확인
2번에서 나온 폴더 경로를 넣는다. `--k 5` = 후보 5개까지.
```bash
python scripts/decode_free.py data/free/NEW3/20261118_xxxxxx_w1_free --k 5 --add-words data/wordlist/korean_password_supplement_dict.csv
```
- 단어마다 **O**(후보 안에 있음) / **X**(없음)와 후보 목록, 맨 아래에 `단어 Top-5 포함 ○/○ = ○%`.
- `--add-words`로 비밀번호 보강 사전을 넣었으므로 인스타·디스코드·배민 같은 단어도 후보에 뜬다.

---

## 참고

### 녹화와 확인을 한 번에
2번에 `--decode`를 붙이면 끝나자마자 바로 결과가 나온다(이때 기본 후보 3개).
```bash
python scripts/record_free.py --subject NEW3 --wear 1 --decode
```

### brute-force용으로 후보를 넓게 뽑기
`--k 20`에 `--csv cand.csv`를 붙이면 단어별 후보 20개가 CSV로 저장된다.
```bash
python scripts/decode_free.py data/free/NEW3/20261118_xxxxxx_w1_free --k 20 --add-words data/wordlist/korean_password_supplement_dict.csv --csv cand.csv
```

### 주요 옵션
| 옵션 | 뜻 |
|---|---|
| `--k N` | 후보를 N개까지 (기본 3) |
| `--add-words 파일` | 사전에 단어 추가(`.csv`는 '단어' 열, `.txt`는 한 줄에 하나). 쌍자음·ㅒ·ㅖ 단어는 자동 제외 |
| `--csv 파일` | 단어별 결과를 CSV로 저장 |
| `--flip -1` | 같은 종류의 링이면 손잡이 고정(단어가 적을 때 권장) |
| `--k 5` 등 | Top-5 성능을 볼 때 |

### 알아둘 점
- **키 로그가 함께 저장돼야 성능(O/X)이 나온다.** 녹화 기본값이 저장이고, 추론에는 쓰지 않고 채점에만 쓴다.
- **사전에 없는 단어**(이름 등)는 단어 후보로 안 뜨고 자음열 후보로만 나온다 → `--add-words`에 넣어야 단어로 복원된다.
- 오타를 내고 백스페이스로 고친 단어는 "채점 제외"로 표시된다.
- 녹화 창은 실제 링으로 검증하지 못했다. 창이 안 뜨거나 오류가 나면 화면 내용을 확인한다.
