// Map pins to your ESP32 configuration
const int dirPin = 32;  // DIR pin on ESP32
const int stepPin = 33; // STEP pin on ESP32

void setup() {
  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
  
  // pause before continuing with main program
  delay(2000);
  
  // set stepper motor direction to clockwise
  digitalWrite(dirPin, HIGH);
}

void loop() {
  // take one step
  digitalWrite(stepPin, HIGH);
  //delayMicroseconds(5000);
  delay(250);
  
  // pause before taking next step
  digitalWrite(stepPin, LOW);
  //delayMicroseconds(5000);
  delay(250);
}