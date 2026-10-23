// Optimized Relay Control Code

const int openTime = 50;

// Define relay pins in the order they are triggered
const int relayPins[] = {
  // 14, // relayPin9 - A0
  // 4,  // relayPin2
  // 5,  // relayPin3
  // 6,  // relayPin4
  // 7,  // relayPin5
  // 8,  // relayPin6
  // 9,  // relayPin7
  // 10, // relayPin8
  // 3,  // relayPin13
  // 12, // relayPin10
  // 13, // relayPin11
  // 2   // relayPin12
};

const int relayCount = sizeof(relayPins) / sizeof(relayPins[0]);

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < relayCount; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW);
    delay(10);
  }
}

void loop() {
  delay(2000);  // Initial delay

  for (int i = 0; i < relayCount; i++) {
    digitalWrite(relayPins[i], HIGH);
    delay(openTime);
    digitalWrite(relayPins[i], LOW);
    delay(1000);
  }
}
