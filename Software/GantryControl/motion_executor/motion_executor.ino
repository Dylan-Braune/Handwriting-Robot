/*
  motion_executor.ino

  ESP32 side of the gantry motion system. This firmware makes NO planning
  decisions -- it only deserializes fixed-size binary frames from the SBC
  (see motion_planner.py's serialize_block/serialize_plan for the exact
  layout) and executes them:

    MOVE frame -> run a Bresenham tick loop for the given signed step counts
                  and per-tick delay (all three numbers came pre-computed
                  from the SBC; this firmware does not do Douglas-Peucker,
                  does not do mm-to-steps conversion, and does not do feed
                  rate math -- it only counts ticks and toggles pins).
    PEN frame  -> actuate the pen-lift motor for a fixed duration.
    END frame  -> stop and wait for the next plan.

  Frame layout (little-endian), mirrors motion_planner.py exactly:
    byte 0        : command (0x01=MOVE, 0x02=PEN, 0x03=END)
    MOVE payload  : int32 stepsX, int32 stepsY, uint32 stepIntervalUs
    PEN payload   : uint8 down (1/0)
    last byte     : XOR checksum of every byte before it (incl. command)

  Extend serial reads are blocking on purpose -- this board does nothing
  else, so there's no reason to make this async/interrupt-driven yet. If
  step rates get high enough that blocking reads start costing accuracy,
  move MOVE-frame reception to happen while the PREVIOUS move is still
  ticking (double-buffer one block ahead) rather than adding any planning
  logic here.
*/

#include <Arduino.h>

// ---- Pin mapping (from your existing sketch_LocatingMaxSpeed.ino test) ----
const int X_DIR_PIN = 32;
const int X_STEP_PIN = 33;
// TODO: wire up + confirm these once the second axis is on the gantry.
const int Y_DIR_PIN = 25;
const int Y_STEP_PIN = 26;

// Pen-lift motor: no driver IC, just an on/off drive signal (e.g. through a
// single MOSFET/transistor switching the motor's supply). TODO: confirm this
// is the right control scheme for your actual motor before relying on it --
// if it's a brushed DC gearmotor you may need a flyback diode across it.
const int PEN_MOTOR_PIN = 27;

// TODO: measure this on the real pen-lift mechanism. Open-loop timing WILL
// drift over hundreds of actuations across a page -- if characters start
// drifting from full pen-down/up partway through a long job, this is the
// first thing to suspect. A cheap fix if that happens: add a microswitch or
// slotted-disc + IR gate at each of the two rest positions and have this
// loop hold the motor on until the switch trips instead of trusting a fixed
// delay -- still zero "planning", just a different stopping condition.
const unsigned long PEN_ACTUATE_MS = 150;

// ---- Protocol constants (must match motion_planner.py) ----
const uint8_t CMD_MOVE = 0x01;
const uint8_t CMD_PEN = 0x02;
const uint8_t CMD_END = 0x03;

void setup() {
  Serial.begin(115200);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT);
  pinMode(PEN_MOTOR_PIN, OUTPUT);
  digitalWrite(PEN_MOTOR_PIN, LOW);
  Serial.println("motion_executor ready");
}

void loop() {
  uint8_t cmd;
  if (!readByte(cmd)) return;

  switch (cmd) {
    case CMD_MOVE: {
      int32_t stepsX, stepsY;
      uint32_t stepIntervalUs;
      uint8_t buf[9];
      if (!readBytes(buf, 9)) return;
      memcpy(&stepsX, buf, 4);
      memcpy(&stepsY, buf + 4, 4);
      memcpy(&stepIntervalUs, buf + 8, 4);

      uint8_t checksum;
      if (!readByte(checksum)) return;
      uint8_t body[10] = { cmd };
      memcpy(body + 1, buf, 9);
      if (xorChecksum(body, 10) != checksum) {
        Serial.println("CHECKSUM ERROR (MOVE)");
        return;
      }
      executeMove(stepsX, stepsY, stepIntervalUs);
      break;
    }
    case CMD_PEN: {
      uint8_t down;
      if (!readByte(down)) return;
      uint8_t checksum;
      if (!readByte(checksum)) return;
      uint8_t body[2] = { cmd, down };
      if (xorChecksum(body, 2) != checksum) {
        Serial.println("CHECKSUM ERROR (PEN)");
        return;
      }
      executePen(down != 0);
      break;
    }
    case CMD_END: {
      uint8_t checksum;
      if (!readByte(checksum)) return;
      uint8_t body[1] = { cmd };
      if (xorChecksum(body, 1) != checksum) {
        Serial.println("CHECKSUM ERROR (END)");
        return;
      }
      // Nothing to do -- job's finished, just wait for the next plan.
      break;
    }
    default:
      Serial.println("UNKNOWN COMMAND BYTE");
      break;
  }
}

// ---- Bresenham execution: identical logic to motion_simulator.py's
// ---- _bresenham_ticks, so what you saw in the simulator plot is exactly
// ---- what this loop will physically draw. ----
void executeMove(int32_t stepsX, int32_t stepsY, uint32_t stepIntervalUs) {
  int32_t ax = abs(stepsX), ay = abs(stepsY);
  int8_t sx = stepsX > 0 ? 1 : -1;
  int8_t sy = stepsY > 0 ? 1 : -1;
  digitalWrite(X_DIR_PIN, sx > 0 ? HIGH : LOW);
  digitalWrite(Y_DIR_PIN, sy > 0 ? HIGH : LOW);

  if (ax >= ay) {
    int32_t err = ax / 2;
    for (int32_t i = 0; i < ax; i++) {
      pulse(X_STEP_PIN, stepIntervalUs);
      err -= ay;
      if (err < 0) {
        pulse(Y_STEP_PIN, stepIntervalUs);
        err += ax;
      }
    }
  } else {
    int32_t err = ay / 2;
    for (int32_t i = 0; i < ay; i++) {
      pulse(Y_STEP_PIN, stepIntervalUs);
      err -= ax;
      if (err < 0) {
        pulse(X_STEP_PIN, stepIntervalUs);
        err += ay;
      }
    }
  }
}

void pulse(int pin, uint32_t intervalUs) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(intervalUs / 2);
  digitalWrite(pin, LOW);
  delayMicroseconds(intervalUs / 2);
}

void executePen(bool down) {
  // Single-direction ~90 degree actuation per call, per your description of
  // the mechanism (a cam/eccentric alternates pen height each quarter turn).
  // Both pen-down and pen-up calls just run the motor the same way -- the
  // mechanical cam is what decides whether that turn raises or lowers the
  // pen next, not this code.
  digitalWrite(PEN_MOTOR_PIN, HIGH);
  delay(PEN_ACTUATE_MS);
  digitalWrite(PEN_MOTOR_PIN, LOW);
}

// ---- small serial helpers ----
bool readByte(uint8_t &out) {
  while (!Serial.available()) { /* block -- see file header note */ }
  out = Serial.read();
  return true;
}

bool readBytes(uint8_t *buf, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!readByte(buf[i])) return false;
  }
  return true;
}

uint8_t xorChecksum(const uint8_t *data, size_t n) {
  uint8_t c = 0;
  for (size_t i = 0; i < n; i++) c ^= data[i];
  return c;
}
