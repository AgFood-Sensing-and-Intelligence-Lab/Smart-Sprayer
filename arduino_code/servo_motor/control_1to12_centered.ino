#include <Servo.h>

Servo servos[12];
const int servoPins[12] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
int servoOffsets[12] = {0, 5, 3, -7, -5, 0, 0, -5, 5, -8, 0, -5}; // Offsets per servo

const int operation_interval = 500;

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 12; i++) {
    servos[i].attach(servoPins[i]);
  }
  for (int i = 0; i < 12; i++) {
    int target_pos = 120;
    if (i%2==0)
      target_pos = 60;
    int angle = servoOffsets[i] + target_pos;
    angle = constrain(angle, 35, 145);
    servos[i].write(angle);
    delay(5);
  }
  Serial.println("All servos reset to default positions.");
  Serial.println("Setup complete");
}

int getServoIndex(char id) {
  if (id >= '1' && id <= '9') return id - '1';       // '1' to '9' → 0 to 8
  if (id == 'a') return 9;
  if (id == 'b') return 10;
  if (id == 'c') return 11;
  return -1; // invalid
}

void loop() {

  delay(1000);
}

