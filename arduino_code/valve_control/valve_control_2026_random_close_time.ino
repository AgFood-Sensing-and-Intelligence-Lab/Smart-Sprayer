const int relayPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
// const int relayPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, A0, A1, A2};
const int totalValves = sizeof(relayPins) / sizeof(relayPins[0]);

char inputBuffer[16];
uint8_t bufferIndex = 0;
bool stringComplete = false;

bool valveStates[totalValves] = {false};
unsigned long valveOpenTime[totalValves] = {0};

// NEW: Array to store the unique random delay for each valve
unsigned long valveDelay[totalValves] = {0};

// NEW: Define your center target and variance
const long CENTER_DELAY = 500;   // Center number in milliseconds
const long DELAY_VARIANCE = 100; // Will generate a delay between 15ms and 35ms

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Seed the random generator using an unconnected analog pin (A3)
  randomSeed(analogRead(A3));
  
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
          
          if (action == 1) {
            valveOpenTime[idx] = millis();
            // NEW: Generate the random delay. 
            // Note: The max value in random() is exclusive, so we add 1.
            long minDelay = CENTER_DELAY - DELAY_VARIANCE;
            long maxDelay = CENTER_DELAY + DELAY_VARIANCE + 1;
            valveDelay[idx] = random(minDelay, maxDelay);
          } else {
            valveOpenTime[idx] = 0;
          }
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
      // NEW: Check against the unique randomized delay for this specific valve
      if (currentMillis - valveOpenTime[i] >= valveDelay[i]) {
        digitalWrite(relayPins[i], LOW);
        valveStates[i] = false;
        valveOpenTime[i] = 0;
        
        Serial.println(i + 1); 
      }
    }
  }
}