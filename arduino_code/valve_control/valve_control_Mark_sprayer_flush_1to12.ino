// Relay pins
const int relayPins[] = {
  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
};

const int relayCount = sizeof(relayPins) / sizeof(relayPins[0]);

const int groupSize = 6;
const int openTime = 500;   // Relay ON time (ms)
const int groupDelay = 1000; // Delay between groups (ms)

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < relayCount; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW);
  }
}

void loop() {

  delay(1000);  // Initial delay

  // Process 4 relays at a time
  for (int start = 0; start < relayCount; start += groupSize) {

    // Open 4 relays simultaneously
    for (int i = start; i < start + groupSize && i < relayCount; i++) {
      digitalWrite(relayPins[i], HIGH);
    }

    Serial.print("Opening relays ");
    Serial.print(start + 1);
    Serial.print(" to ");
    Serial.println(start + groupSize);

    // Keep them open
    delay(openTime);

    // Close the same 4 relays simultaneously
    for (int i = start; i < start + groupSize && i < relayCount; i++) {
      digitalWrite(relayPins[i], LOW);
    }

    Serial.print("Closing relays ");
    Serial.print(start + 1);
    Serial.print(" to ");
    Serial.println(start + groupSize);

    delay(groupDelay);
  }
}