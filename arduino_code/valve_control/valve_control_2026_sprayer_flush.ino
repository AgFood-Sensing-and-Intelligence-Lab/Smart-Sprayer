// Optimized Relay Control Code

// const int openTime = 500;  // 10 valves support, and the other 2 valves cannot work
const int openTime = 5000;  // 10 valves support, and the other 2 valves cannot work

// Define relay pins in the order they are triggered
// const int relayPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
const int relayPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, A0, A1, A2};

const int relayCount = sizeof(relayPins) / sizeof(relayPins[0]);

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < relayCount; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW);
    delay(10);
  }
}

void loop() {
  delay(3000);  // Initial delay

  for (int i = 0; i < relayCount; i++) {
    digitalWrite(relayPins[i], HIGH);
    delay(openTime);
    digitalWrite(relayPins[i], LOW);
    delay(1000);
  }
  // delay(50000);  // Initial delay
}
