const int relayPins[] = {
  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
};

const int totalValves = sizeof(relayPins) / sizeof(relayPins[0]);

String inputString = "";
bool stringComplete = false;

// Track valve states: true = open, false = closed
bool valveStates[totalValves] = {false};
// Track the time (millis) when each valve was opened
unsigned long valveOpenTime[totalValves] = {0};
// Auto-close delay in milliseconds
const unsigned long AUTO_CLOSE_DELAY = 150;

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for serial connection (for Leonardo/Micro)

  for (int i = 0; i < totalValves; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW); // Start closed
  }

  inputString.reserve(16);
}

void loop() {
  if (stringComplete) {
    inputString.trim();

    if (inputString == "0") {
      // Close all valves
      for (int i = 0; i < totalValves; i++) {
        digitalWrite(relayPins[i], LOW);
        valveStates[i] = false;
        valveOpenTime[i] = 0;
      }
      Serial.println("All valves closed.");
    } else {
      int dashIndex = inputString.indexOf('-');

      if (dashIndex > 0) {
        int valveNum = inputString.substring(0, dashIndex).toInt();
        int action = inputString.substring(dashIndex + 1).toInt();

        if (valveNum >= 1 && valveNum <= totalValves && (action == 0 || action == 1)) {
          int idx = valveNum - 1;
          int relayPin = relayPins[idx];
          digitalWrite(relayPin, action == 1 ? HIGH : LOW);
          valveStates[idx] = (action == 1);

          if (action == 1) {
            valveOpenTime[idx] = millis();  // Record when valve was opened
            Serial.print("Valve ");
            Serial.print(valveNum);
            Serial.println(" opened (auto-close in 0.15s)");
          } else {
            valveOpenTime[idx] = 0; // Reset timer on close
            Serial.print("Valve ");
            Serial.print(valveNum);
            Serial.println(" closed");
          }
        } else {
          Serial.println("Invalid valve number or action.");
        }
      } else {
        Serial.println("Invalid format. Use e.g., 5-1 or 8-0");
      }
    }

    inputString = "";
    stringComplete = false;
  }

  // Auto-close valves after 0.3 seconds
  unsigned long currentMillis = millis();
  for (int i = 0; i < totalValves; i++) {
    if (valveStates[i] && valveOpenTime[i] > 0) {
      if (currentMillis - valveOpenTime[i] >= AUTO_CLOSE_DELAY) {
        digitalWrite(relayPins[i], LOW);
        valveStates[i] = false;
        valveOpenTime[i] = 0;
        Serial.print("Valve ");
        Serial.print(i + 1);
        Serial.println(" auto-closed after 0.3s");
      }
    }
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
