const int relayPins[] = {
  14, // valve 1 -> A0
   4, // valve 2
   5, // valve 3
   6, // valve 4
   7, // valve 5
   8, // valve 6
   9, // valve 7
  10, // valve 8
   3, // valve 9
  12, // valve 10
  13, // valve 11
   2  // valve 12
};

const int totalValves = sizeof(relayPins) / sizeof(relayPins[0]);

String inputString = "";
bool stringComplete = false;

// Track valve states: true=open, false=closed
bool valveStates[totalValves] = {false};
// Track when valve was opened (millis)
unsigned long valveOpenTime[totalValves] = {0};
// Auto-close delay in ms
const unsigned long AUTO_CLOSE_DELAY = 300;

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for serial (Leonardo/Micro)

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
            valveOpenTime[idx] = millis();  // Record open time
            Serial.print("Valve ");
            Serial.print(valveNum);
            Serial.println(" opened (auto-close in 0.3s)");
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

  // Check for valves to auto-close
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
