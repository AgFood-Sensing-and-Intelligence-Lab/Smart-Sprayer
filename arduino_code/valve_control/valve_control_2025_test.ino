// Optimized Relay Control Code

// const int openTime = 10; # can be lower to 10ms for Mark' sprayer
const int openTime = 50;

// Define relay pins in the order they are triggered
const int relayPins[] = {
  2,  
  3,  
  4,  
  5,  
  6,  
  7,  
  8,  
  9,  
  10, 
  11, 
  12, 
  13  
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
  delay(3000);  // Initial delay

  for (int i = 0; i < relayCount; i++) {
    digitalWrite(relayPins[i], HIGH);
    delay(openTime);
    digitalWrite(relayPins[i], LOW);
    delay(1000);
  }
}
