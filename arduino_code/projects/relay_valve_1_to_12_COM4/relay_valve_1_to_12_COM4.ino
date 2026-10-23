// Relay Control with Dash-Separated Serial Commands (e.g., "1-1" to open valve 1)

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

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for serial (optional for Leonardo/Micro)

  for (int i = 0; i < totalValves; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW); // Ensure all valves start closed
  }

  inputString.reserve(16); // Reserve buffer memory
}

void loop() {
  if (stringComplete) {
    inputString.trim(); // Remove any trailing whitespace/newlines

    // If input is just "0", close all valves
    if (inputString == "0") {
      for (int i = 0; i < totalValves; i++) {
        digitalWrite(relayPins[i], LOW);
      }
      Serial.println("All valves closed.");
    } else {
      int dashIndex = inputString.indexOf('-');

      if (dashIndex > 0) {
        int valveNum = inputString.substring(0, dashIndex).toInt();
        int action = inputString.substring(dashIndex + 1).toInt();

        if (valveNum >= 1 && valveNum <= totalValves && (action == 0 || action == 1)) {
          int relayPin = relayPins[valveNum - 1];
          digitalWrite(relayPin, action == 1 ? HIGH : LOW);

          Serial.print("Valve ");
          Serial.print(valveNum);
          Serial.println(action == 1 ? " opened" : " closed");
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
