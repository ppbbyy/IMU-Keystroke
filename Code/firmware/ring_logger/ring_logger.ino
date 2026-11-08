// ring_logger — Arduino Nano 33 BLE (내장 LSM9DS1) 6축(+자력계) USB 시리얼 스트리머
//
// 필요 라이브러리: Arduino_LSM9DS1 (라이브러리 매니저)
//
// 프로토콜 (115200 8N1, 줄 단위 ASCII; USB CDC라 실제 속도는 USB 속도)
//   호스트 -> 장치:  'G' 스트리밍 시작(시퀀스 0부터),  'X' 중지,  'I' 정보 재전송
//   장치 -> 호스트:
//     H,<fw>,<acc_hz>,<gyro_hz>,<mag_hz>      헤더(부팅 시, 'I' 응답 시)
//     D,<seq>,<t_us>,<ax_mg>,<ay_mg>,<az_mg>,<gx_mdps>,<gy_mdps>,<gz_mdps>,<mx_nT>,<my_nT>,<mz_nT>,<mag_new>
//       seq      : 샘플 번호(uint32), 손실 검출용
//       t_us     : micros() (uint32, 약 71.6분마다 래핑 -> 호스트가 언래핑)
//       acc      : 밀리 g,  gyro: 밀리 deg/s,  mag: 나노 tesla (정수; 호스트가 단위 변환)
//       mag_new  : 1이면 이 줄에서 자력계가 새로 갱신됨, 0이면 직전 값 반복(자력계는 약 20Hz)
//
// 가속도/자이로는 둘 다 준비된 시점에 한 줄로 내보낸다(두 센서 ODR 동일, 라이브러리 기본 119Hz).
// 축 방향은 Arduino_LSM9DS1 라이브러리가 돌려주는 값 그대로다(보드 실크 기준 변환 없음).

#include <Arduino_LSM9DS1.h>

#define FW_VERSION "ringkey-1.0"

static bool streaming = false;
static uint32_t seq = 0;
static float mx = 0, my = 0, mz = 0;

static void printHeader() {
  Serial.print("H,");
  Serial.print(FW_VERSION);
  Serial.print(",");
  Serial.print(IMU.accelerationSampleRate(), 1);
  Serial.print(",");
  Serial.print(IMU.gyroscopeSampleRate(), 1);
  Serial.print(",");
  Serial.println(IMU.magneticFieldSampleRate(), 1);
}

static void drainIMU() {
  float a, b, c;
  while (IMU.accelerationAvailable()) IMU.readAcceleration(a, b, c);
  while (IMU.gyroscopeAvailable()) IMU.readGyroscope(a, b, c);
  while (IMU.magneticFieldAvailable()) IMU.readMagneticField(a, b, c);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }
  if (!IMU.begin()) {
    while (true) {
      Serial.println("E,imu_init_failed");
      delay(1000);
    }
  }
  printHeader();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'G') {
      drainIMU();  // 오래된 샘플 버림
      seq = 0;
      streaming = true;
    } else if (c == 'X') {
      streaming = false;
    } else if (c == 'I') {
      printHeader();
    }
  }
  if (!streaming) return;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    uint32_t t = micros();
    float ax, ay, az, gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    int magNew = 0;
    if (IMU.magneticFieldAvailable()) {
      IMU.readMagneticField(mx, my, mz);
      magNew = 1;
    }

    char buf[160];
    int n = snprintf(buf, sizeof(buf),
                     "D,%lu,%lu,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%d\n",
                     (unsigned long)seq, (unsigned long)t,
                     lroundf(ax * 1000.0f), lroundf(ay * 1000.0f), lroundf(az * 1000.0f),
                     lroundf(gx * 1000.0f), lroundf(gy * 1000.0f), lroundf(gz * 1000.0f),
                     lroundf(mx * 1000.0f), lroundf(my * 1000.0f), lroundf(mz * 1000.0f),
                     magNew);
    Serial.write((const uint8_t *)buf, n);
    seq++;
  }
}
