// Jarvis pan-tilt: Wemos D1 mini + 2x SG90, controlled over USB serial (115200, lines ending with \n).
//
// Commands:
//   P <deg>                         PAN (left-right) to an angle
//   T <deg>                         TILT (up-down) to an angle
//   M <pan> <tilt>                  both; one after the other (PAN first), so the servo currents do not add up
//   C                               center (90 90)
//   ?                               position, replies right away
//   L                               show limits
//   L <pmin> <pmax> <tmin> <tmax>   set limits (only until restart, for calibration)
//
// Angles outside the limits are clamped. A move ends with the reply "OK P=<pan> T=<tilt>",
// a move command during a move gets "ERR busy". After boot: "READY P=90 T=90".

#include <ESP8266WiFi.h>
#include <Servo.h>

const uint8_t PAN_PIN = D5;
const uint8_t TILT_PIN = D6;
const uint16_t PULSE_MIN_US = 500;   // SG90: ~0°
const uint16_t PULSE_MAX_US = 2400;  // SG90: ~180°
const unsigned long STEP_MS = 15;    // 1° every 15 ms, about 65°/s
const int CENTER = 90;

// Cautious limits until the stand is calibrated.
int panMin = 30, panMax = 150;
int tiltMin = 45, tiltMax = 135;

Servo pan, tilt;
int panPos = CENTER, tiltPos = CENTER;
int panTarget = CENTER, tiltTarget = CENTER;
bool moving = false;
unsigned long lastStep = 0;

char buf[48];
uint8_t len = 0;

void report(const char *prefix) {
  Serial.printf("%s P=%d T=%d\n", prefix, panPos, tiltPos);
}

void startMove(int p, int t) {
  if (moving) {
    Serial.println("ERR busy");
    return;
  }
  panTarget = constrain(p, panMin, panMax);
  tiltTarget = constrain(t, tiltMin, tiltMax);
  moving = true;
}

void handleLimits(const char *args) {
  int a, b, c, d;
  int n = sscanf(args, "%d %d %d %d", &a, &b, &c, &d);
  if (n == 4) {
    if (a < 0 || b > 180 || a >= b || c < 0 || d > 180 || c >= d) {
      Serial.println("ERR limits must be 0 <= min < max <= 180");
      return;
    }
    panMin = a; panMax = b; tiltMin = c; tiltMax = d;
  } else if (n > 0) {
    Serial.println("ERR usage: L <pmin> <pmax> <tmin> <tmax>");
    return;
  }
  Serial.printf("LIM P=%d..%d T=%d..%d\n", panMin, panMax, tiltMin, tiltMax);
}

void handle(char *cmd) {
  int a, b;
  switch (cmd[0]) {
    case 'P': case 'p':
      if (sscanf(cmd + 1, "%d", &a) == 1) startMove(a, tiltPos);
      else Serial.println("ERR usage: P <deg>");
      break;
    case 'T': case 't':
      if (sscanf(cmd + 1, "%d", &a) == 1) startMove(panPos, a);
      else Serial.println("ERR usage: T <deg>");
      break;
    case 'M': case 'm':
      if (sscanf(cmd + 1, "%d %d", &a, &b) == 2) startMove(a, b);
      else Serial.println("ERR usage: M <pan> <tilt>");
      break;
    case 'C': case 'c':
      startMove(CENTER, CENTER);
      break;
    case '?':
      Serial.printf("POS P=%d T=%d %s\n", panPos, tiltPos, moving ? "MOVING" : "IDLE");
      break;
    case 'L': case 'l':
      handleLimits(cmd + 1);
      break;
    default:
      Serial.printf("ERR unknown: %s\n", cmd);
  }
}

void readSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      buf[len] = 0;
      if (len) handle(buf);
      len = 0;
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = ch;
    }
  }
}

void setup() {
  // PoC powered from the computer's USB: WiFi off means about 70 mA more for the servos.
  WiFi.mode(WIFI_OFF);
  WiFi.forceSleepBegin();
  Serial.begin(115200);
  // Servos are attached with a gap, so their jumps to center do not overlap in current draw.
  pan.attach(PAN_PIN, PULSE_MIN_US, PULSE_MAX_US, CENTER);
  delay(400);
  tilt.attach(TILT_PIN, PULSE_MIN_US, PULSE_MAX_US, CENTER);
  delay(400);
  Serial.println();
  report("READY");
}

void loop() {
  readSerial();
  if (!moving || millis() - lastStep < STEP_MS) return;
  lastStep = millis();
  if (panPos != panTarget) {
    panPos += panTarget > panPos ? 1 : -1;
    pan.write(panPos);
  } else if (tiltPos != tiltTarget) {
    tiltPos += tiltTarget > tiltPos ? 1 : -1;
    tilt.write(tiltPos);
  } else {
    moving = false;
    report("OK");
  }
}
