// Optimized Relay Control Code

const int openTimes[] = {100, 500, 1000};
const int totalLoops = sizeof(openTimes) / sizeof(openTimes[0]);


// row close to camera, COM7
// const int relayPins[] = {
//   2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
// };
// row far to camera, COM8
const int relayPins[] = {
  2, 3, 4, 5, 6, 7, 8, 9, 10, A0, A1, A2
};

const int relayCount = sizeof(relayPins) / sizeof(relayPins[0]);

int loopIndex = 0;


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

  int openTime = openTimes[loopIndex];

  Serial.print("Testing open time: ");
  Serial.print(openTime);
  Serial.println(" ms");


  for (int i = 0; i < relayCount; i++) {

    digitalWrite(relayPins[i], HIGH);

    delay(openTime);

    digitalWrite(relayPins[i], LOW);

    delay(1000);  // Interval between valves
  }


  // Move to next open time
  loopIndex++;

  if (loopIndex >= totalLoops) {
    loopIndex = 0;
  }
}