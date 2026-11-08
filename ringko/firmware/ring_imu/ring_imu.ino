// ring_imu — Arduino Nano 33 BLE (rev1, 내장 LSM9DS1) IMU 스트리머
//
// 라이브러리 불필요(Wire1로 레지스터 직접 설정). Nano 33 BLE "Rev2"는 BMI270이라 동작하지 않음.
//
// 설정: 가속도+자이로 238Hz(동기), 가속도 ±8g, 자이로 ±2000dps, 자력계 80Hz ±4gauss
//   - 238Hz: QwertyRing은 200Hz 이상에서 터치 검출이 유지됨, aLeak은 30Hz까지 유지. 97Hz보다 여유 있게.
//   - ±8g: 키 타격 순간 스파이크가 ±4g를 넘을 수 있어 포화 방지.
//
// 시리얼(USB CDC, 줄 단위 ASCII)
//   호스트 -> 장치: 'G' 스트리밍 시작(seq=0)  'X' 중지  'I' 헤더 요청  'P' 시계 핑
//   장치 -> 호스트:
//     H,<fw>,<odr_hz>,<acc_mg_per_lsb>,<gyro_mdps_per_lsb>,<mag_mgauss_per_lsb>,hfxo=<0|1>
//     D,<seq>,<t_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>      (원시 int16, 단위 변환은 호스트가 헤더 감도로)
//     M,<t_us>,<mx>,<my>,<mz>                          (자력계, 원시 int16)
//     P,<t_us>                                         (핑 응답: 'P'를 받은 순간의 micros)
//     E,<msg>                                          (오류)
//   t_us = micros() (uint32, 약 71.6분마다 래핑 -> 호스트가 언래핑)
//   축 방향 = LSM9DS1 칩 좌표 그대로.

#include <Wire.h>
#if __has_include("nrf.h")
#include "nrf.h"
#define HAVE_NRF_CLOCK 1
#endif

#define FW_VERSION "ringko-fw-2"
// fw-2: 32MHz 크리스털(HFXO) 강제 시작. 기본 내부 RC(HFINT)로는 micros()가 ~1.6% 틀리고 흔들림(파일럿 실측).

static const uint8_t AG = 0x6B;   // 가속도/자이로
static const uint8_t MAG = 0x1E;  // 자력계

// AG 레지스터
static const uint8_t WHO_AM_I = 0x0F;      // AG: 0x68, MAG: 0x3D
static const uint8_t CTRL_REG1_G = 0x10;
static const uint8_t CTRL_REG6_XL = 0x20;
static const uint8_t CTRL_REG8 = 0x22;
static const uint8_t STATUS_REG = 0x17;    // bit1 GDA, bit0 XLDA
static const uint8_t OUT_X_L_G = 0x18;
static const uint8_t OUT_X_L_XL = 0x28;
// MAG 레지스터
static const uint8_t CTRL_REG1_M = 0x20;
static const uint8_t CTRL_REG2_M = 0x21;
static const uint8_t CTRL_REG3_M = 0x22;
static const uint8_t CTRL_REG4_M = 0x23;
static const uint8_t STATUS_REG_M = 0x27;  // bit3 ZYXDA
static const uint8_t OUT_X_L_M = 0x28;

// 데이터시트 감도(LSM9DS1 Table 3)
static const float ODR_HZ = 238.0f;
static const float ACC_MG_PER_LSB = 0.244f;    // ±8g
static const float GYRO_MDPS_PER_LSB = 70.0f;  // ±2000dps
static const float MAG_MGAUSS_PER_LSB = 0.14f; // ±4gauss

static bool streaming = false;
static uint32_t seq = 0;
static uint32_t lastMagCheck = 0;

static void wr(uint8_t dev, uint8_t reg, uint8_t v) {
  Wire1.beginTransmission(dev);
  Wire1.write(reg);
  Wire1.write(v);
  Wire1.endTransmission();
}

static int rd(uint8_t dev, uint8_t reg) {
  Wire1.beginTransmission(dev);
  Wire1.write(reg);
  if (Wire1.endTransmission(false) != 0) return -1;
  if (Wire1.requestFrom(dev, (uint8_t)1) != 1) return -1;
  return Wire1.read();
}

static bool rdn(uint8_t dev, uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire1.beginTransmission(dev);
  Wire1.write(0x80 | reg);  // MSB=1: 자동 증가(자력계 필수, AG는 IF_ADD_INC로도 동작)
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(dev, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire1.read();
  return true;
}

static inline int16_t i16(const uint8_t *b) { return (int16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8)); }

static void startHfxo() {
#ifdef HAVE_NRF_CLOCK
  NRF_CLOCK->EVENTS_HFCLKSTARTED = 0;
  NRF_CLOCK->TASKS_HFCLKSTART = 1;
  uint32_t t0 = millis();
  while (!NRF_CLOCK->EVENTS_HFCLKSTARTED && millis() - t0 < 200) {
  }
#endif
}

static bool hfxoRunning() {
#ifdef HAVE_NRF_CLOCK
  // HFCLKSTAT: bit16 STATE(1=실행 중), bit0 SRC(1=크리스털)
  return (NRF_CLOCK->HFCLKSTAT & 0x10001) == 0x10001;
#else
  return false;
#endif
}

static void printHeader() {
  Serial.print("H," FW_VERSION ",");
  Serial.print(ODR_HZ, 1);
  Serial.print(",");
  Serial.print(ACC_MG_PER_LSB, 3);
  Serial.print(",");
  Serial.print(GYRO_MDPS_PER_LSB, 3);
  Serial.print(",");
  Serial.print(MAG_MGAUSS_PER_LSB, 3);
  Serial.print(",hfxo=");
  Serial.println(hfxoRunning() ? 1 : 0);
}

static bool imuInit(int &agId, int &mId) {
  wr(AG, CTRL_REG8, 0x05);    // SW_RESET + IF_ADD_INC
  wr(MAG, CTRL_REG2_M, 0x0C); // REBOOT + SOFT_RST
  delay(20);
  agId = rd(AG, WHO_AM_I);
  mId = rd(MAG, WHO_AM_I);
  if (agId != 0x68 || mId != 0x3D) return false;
  wr(AG, CTRL_REG8, 0x44);     // BDU + IF_ADD_INC
  wr(AG, CTRL_REG1_G, 0x98);   // ODR 238Hz(100), FS 2000dps(11), BW 00
  wr(AG, CTRL_REG6_XL, 0x98);  // ODR 238Hz(100), FS ±8g(11)
  wr(MAG, CTRL_REG1_M, 0x7C);  // XY ultra-high perf(11), ODR 80Hz(111)
  wr(MAG, CTRL_REG2_M, 0x00);  // ±4 gauss
  wr(MAG, CTRL_REG3_M, 0x00);  // 연속 변환
  wr(MAG, CTRL_REG4_M, 0x0C);  // Z ultra-high perf
  return true;
}

void setup() {
  startHfxo();
  Serial.begin(115200);
  while (!Serial) {
  }
#ifdef PIN_ENABLE_SENSORS_3V3
  pinMode(PIN_ENABLE_SENSORS_3V3, OUTPUT);
  digitalWrite(PIN_ENABLE_SENSORS_3V3, HIGH);
#endif
#ifdef PIN_ENABLE_I2C_PULLUP
  pinMode(PIN_ENABLE_I2C_PULLUP, OUTPUT);
  digitalWrite(PIN_ENABLE_I2C_PULLUP, HIGH);
#endif
  delay(10);
  Wire1.begin();
  Wire1.setClock(400000);
  int agId, mId;
  while (!imuInit(agId, mId)) {
    Serial.print("E,imu_init_failed whoami_ag=");
    Serial.print(agId);
    Serial.print(" whoami_m=");
    Serial.print(mId);
    Serial.println(" (rev1 LSM9DS1 expected: 104/61)");
    delay(1000);
  }
  printHeader();
}

void loop() {
  while (Serial.available()) {
    int c = Serial.read();
    if (c == 'P') {
      uint32_t t = micros();
      Serial.print("P,");
      Serial.println(t);
    } else if (c == 'G') {
      seq = 0;
      streaming = true;
    } else if (c == 'X') {
      streaming = false;
    } else if (c == 'I') {
      printHeader();
    }
  }
  if (!streaming) return;

  int st = rd(AG, STATUS_REG);
  if (st >= 0 && (st & 0x02)) {  // 자이로 새 샘플(가속도와 동기)
    uint32_t t = micros();
    uint8_t g[6], a[6];
    if (rdn(AG, OUT_X_L_G, g, 6) && rdn(AG, OUT_X_L_XL, a, 6)) {
      char buf[96];
      int n = snprintf(buf, sizeof(buf), "D,%lu,%lu,%d,%d,%d,%d,%d,%d\n",
                       (unsigned long)seq, (unsigned long)t,
                       i16(a), i16(a + 2), i16(a + 4), i16(g), i16(g + 2), i16(g + 4));
      Serial.write((const uint8_t *)buf, n);
      seq++;
    }
  }

  uint32_t now = micros();
  if (now - lastMagCheck >= 4000) {  // 자력계는 80Hz라 4ms마다만 확인
    lastMagCheck = now;
    int sm = rd(MAG, STATUS_REG_M);
    if (sm >= 0 && (sm & 0x08)) {
      uint8_t m[6];
      if (rdn(MAG, OUT_X_L_M, m, 6)) {
        char buf[64];
        int n = snprintf(buf, sizeof(buf), "M,%lu,%d,%d,%d\n", (unsigned long)micros(), i16(m), i16(m + 2), i16(m + 4));
        Serial.write((const uint8_t *)buf, n);
      }
    }
  }
}
