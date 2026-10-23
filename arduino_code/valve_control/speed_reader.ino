// Wheel Speed and Distance Monitor
// Hall sensor on pin 3, magnet on wheel

const int hallPin = 3;
const int ledPin = 13;

// Wheel specifications
const float wheelDiameter = 45.49;  // mm
const float distancePerRotation = 3.14159 * wheelDiameter;  // mm per rotation (≈142.91 mm)

int hallState = 0;
int lastHallState = 0;
int rotationCount = 0;

unsigned long lastRotationTime = 0;
unsigned long currentTime = 0;
unsigned long startTime = 0;  // Track total elapsed time from first rotation
float totalDistance = 0;  // in mm
float currentSpeed = 0;   // in mm/s

void setup() {
  Serial.begin(9600);
  pinMode(hallPin, INPUT);
  pinMode(ledPin, OUTPUT);
 
  lastHallState = digitalRead(hallPin);
 
  Serial.println("=== WHEEL SPEED MONITOR ===");
  Serial.print("Wheel diameter: ");
  Serial.print(wheelDiameter);
  Serial.println(" mm");
  Serial.print("Distance per rotation: ");
  Serial.print(distancePerRotation, 2);
  Serial.println(" mm");
  Serial.println("Send 'R' or 'r' to reset counters");
  Serial.println("------------------------");
}

void loop() {
  hallState = digitalRead(hallPin);
  currentTime = millis();
 
  // Check for reset command
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'R' || command == 'r') {
      resetCounters();
    }
  }
 
  // Detect rotation (rising edge - magnet approaching sensor)
  if (hallState == HIGH && lastHallState == LOW) {
   
    rotationCount++;
    digitalWrite(ledPin, HIGH);
   
    // Set start time on first rotation
    if (rotationCount == 1) {
      startTime = currentTime;
      lastRotationTime = currentTime;
     
      // Calculate total distance for first rotation
      totalDistance = rotationCount * distancePerRotation;
     
      Serial.println("--- ROTATION DATA ---");
      Serial.print("Rotation #: ");
      Serial.println(rotationCount);
      Serial.print("Distance: ");
      Serial.print(distancePerRotation / 10.0, 2);
      Serial.println(" cm");
      Serial.println("First rotation - timer started!");
      Serial.print("Total Distance: ");
      Serial.print(totalDistance / 10.0, 1);
      Serial.println(" cm");
      Serial.println("--------------------");
     
    } else {
      // Calculate time for this rotation (from previous rotation)
      unsigned long rotationTime = currentTime - lastRotationTime;
     
      // Calculate speed for this rotation (mm/s)
      currentSpeed = distancePerRotation / (rotationTime / 1000.0);
     
      // Calculate total distance
      totalDistance = rotationCount * distancePerRotation;
     
      // Calculate total elapsed time since first rotation
      float totalElapsedTime = (currentTime - startTime) / 1000.0; // seconds
     
      // Print results
      Serial.println("--- ROTATION DATA ---");
      Serial.print("Rotation #: ");
      Serial.println(rotationCount);
     
      Serial.print("Distance: ");
      Serial.print(distancePerRotation / 10.0, 2);  // Convert mm to cm
      Serial.println(" cm");
     
      Serial.print("Time: ");
      Serial.print(rotationTime / 1000.0, 3);  // Convert ms to seconds
      Serial.println(" seconds");
     
      Serial.print("Speed: ");
      Serial.print(currentSpeed / 10.0, 2);  // Convert mm/s to cm/s
      Serial.print(" cm/s (");
      Serial.print((currentSpeed / 10.0) * 0.0224, 4);  // Convert cm/s to mph
      Serial.println(" mph)");
     
      Serial.print("Total Distance: ");
      Serial.print(totalDistance / 10.0, 1);  // Convert mm to cm
      Serial.println(" cm");
     
      // Calculate and display average speed
      float avgSpeed = (totalDistance / 10.0) / totalElapsedTime;  // cm/s
      Serial.print("AVG Speed: ");
      Serial.print(avgSpeed, 2);
      Serial.print(" cm/s (");
      Serial.print(avgSpeed * 0.0224, 4);
      Serial.println(" mph)");
     
      Serial.println("--------------------");
    }
   
    // Update last rotation time
    lastRotationTime = currentTime;
   
  } else if (hallState == LOW && lastHallState == HIGH) {
    // Magnet moved away from sensor
    digitalWrite(ledPin, LOW);
  }
 
  lastHallState = hallState;
 
  delay(10);  // Small delay for stability
}

// Reset function
void resetCounters() {
  rotationCount = 0;
  totalDistance = 0;
  startTime = 0;
  currentSpeed = 0;
  lastRotationTime = 0;
 
  Serial.println();
  Serial.println("*** COUNTERS RESET ***");
  Serial.println("Ready for new measurement session!");
  Serial.println("------------------------");
}