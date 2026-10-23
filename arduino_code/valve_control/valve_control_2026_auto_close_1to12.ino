const int relayPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
const int totalValves = sizeof(relayPins) / sizeof(relayPins[0]);

char inputBuffer[16];
uint8_t bufferIndex = 0;
bool stringComplete = false;

bool valveStates[totalValves] = {false};
unsigned long valveOpenTime[totalValves] = {0};

// Auto-close delay in milliseconds
// const unsigned long AUTO_CLOSE_DELAY = 25;
// const unsigned long AUTO_CLOSE_DELAY = 100;
// const unsigned long AUTO_CLOSE_DELAY = 500;
const unsigned long AUTO_CLOSE_DELAY = 1000;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  for (int i = 0; i < totalValves; i++) {
    pinMode(relayPins[i], OUTPUT);
    digitalWrite(relayPins[i], LOW); // Start closed
  }
}

void loop() {
  // 1. Read Serial Data (Non-blocking)
  while (Serial.available() > 0 && !stringComplete) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0'; // Null-terminate
        stringComplete = true;
      }
    } else if (bufferIndex < 15) {
      inputBuffer[bufferIndex] = c;
      bufferIndex++;
    }
  }

  // 2. Process Command
  if (stringComplete) {
    if (inputBuffer[0] == '0' && inputBuffer[1] == '\0') {
      // Close all valves
      for (int i = 0; i < totalValves; i++) {
        digitalWrite(relayPins[i], LOW);
        valveStates[i] = false;
        valveOpenTime[i] = 0;
      }
    } else {
      // Parse individual valve command
      char* dashPos = strchr(inputBuffer, '-');
      
      if (dashPos != NULL) {
        *dashPos = '\0'; 
        
        int valveNum = atoi(inputBuffer);
        int action = atoi(dashPos + 1);

        if (valveNum >= 1 && valveNum <= totalValves && (action == 0 || action == 1)) {
          int idx = valveNum - 1;
          
          digitalWrite(relayPins[idx], action == 1 ? HIGH : LOW);
          valveStates[idx] = (action == 1);
          valveOpenTime[idx] = (action == 1) ? millis() : 0;
        }
      }
    }

    // Reset buffer
    bufferIndex = 0;
    inputBuffer[0] = '\0';
    stringComplete = false;
  }

  // 3. Auto-close valves
  unsigned long currentMillis = millis();
  for (int i = 0; i < totalValves; i++) {
    if (valveStates[i] && valveOpenTime[i] > 0) {
      if (currentMillis - valveOpenTime[i] >= AUTO_CLOSE_DELAY) {
        digitalWrite(relayPins[i], LOW);
        valveStates[i] = false;
        valveOpenTime[i] = 0;
        
        Serial.println(i + 1); 
      }
    }
  }
}