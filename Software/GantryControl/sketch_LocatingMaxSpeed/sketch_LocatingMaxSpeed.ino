const int dirPin = 32;  // Your DIR pin
const int stepPin = 33; // Your STEP pin

// --- CONFIGURATION VARIABLES ---
const float stepAngle = 1.8; // Standard NEMA 17 step angle
const int stepsPerRev = 360 / stepAngle; // 200 steps for full-step mode

// --- LIVE VARIABLES ---
volatile float targetRPM = 30.0;    // Starting speed default (30 RPM)
volatile unsigned long stepDelayUs = 5000; // Calculated delay in microseconds

void setup() {
  Serial.begin(115200);
  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
  
  digitalWrite(dirPin, HIGH); // Set default direction
  
  // Calculate initial delay for the default 30 RPM
  calculateDelay(targetRPM);
  
  delay(1000);
  printInstructions();
}

void loop() {
  // Check if you typed a new speed into the Serial Monitor
  if (Serial.available() > 0) {
    float inputRPM = Serial.parseFloat();
    
    // Clear any remaining newline or carriage return characters
    while(Serial.available() > 0) { Serial.read(); }

    if (inputRPM > 0.1) {
      targetRPM = inputRPM;
      calculateDelay(targetRPM);
    } else {
      Serial.println("⚠️ Invalid RPM. Enter a number greater than 0.");
    }
  }

  // Continuously spin the motor at the currently requested speed
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(stepDelayUs);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(stepDelayUs);
}

// Function to handle the step calculation math
void calculateDelay(float rpm) {
  // Total time for one single step in microseconds
  float totalStepTimeUs = (60.0 * 1000000.0) / (rpm * (float)stepsPerRev);
  
  // Split the total time between the HIGH pulse and LOW pulse
  stepDelayUs = (unsigned long)(totalStepTimeUs / 2.0);

  Serial.print("🔄 Target Speed Set To: ");
  Serial.print(rpm);
  Serial.print(" RPM | Calculated Pulse Delay: ");
  Serial.print(stepDelayUs);
  Serial.println(" microseconds");
}

void printInstructions() {
  Serial.println("\n=============================================");
  Serial.println("🤖 ESP32 Stepper Speed Tester Ready!");
  Serial.println("Type your desired RPM into the bar above and hit Enter.");
  Serial.println("Example: Type '60' for 1 rev per second.");
  Serial.print("Current starting speed: ");
  Serial.print(targetRPM);
  Serial.println(" RPM");
  Serial.println("=============================================\n");
}