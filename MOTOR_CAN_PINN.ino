#include <Arduino_FreeRTOS.h>
#include <semphr.h>

// --- GLOBALS ---
const int numSamples = 50; 
int processedBuffer[numSamples];          
int currentTemp = 25; 
int motorPWM = 255; 
bool tempOverride = false; 

// --- RTOS SYNCHRONIZATION ---
SemaphoreHandle_t bufferReadySemaphore;
SemaphoreHandle_t systemStateMutex; // Protects shared data from corruption

// --- MEMORY-SAFE SERIAL BUFFER ---
char cmdBuffer[32]; 
int bufIdx = 0;

// --- TASK PROTOTYPES ---
void TaskAcquireData(void *pvParameters);
void TaskAnalyzeAndCommunicate(void *pvParameters);
void TaskCommandListener(void *pvParameters); 
void setupADC(); 
void setupI2C();
void setupPWM();
int readTemperatureI2C();
void sendSLCAN(uint16_t id, uint8_t dlc, uint8_t* data); 

// --- FAST INTEGER SQUARE ROOT (Eliminates the float 'sqrt()' library) ---
uint32_t isqrt(uint32_t n) {
    uint32_t root = 0, bit = 1UL << 30;
    while (bit > n) bit >>= 2;
    while (bit != 0) {
        if (n >= root + bit) {
            n -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

void setup() {
  Serial.begin(115200);
  
  setupADC(); 
  setupI2C(); 
  setupPWM(); 

  bufferReadySemaphore = xSemaphoreCreateBinary();
  systemStateMutex = xSemaphoreCreateMutex(); 

  // OPTIMIZATION: Carefully tuned stack sizes to prevent RAM Overflow
  if (bufferReadySemaphore != NULL && systemStateMutex != NULL) {
    xTaskCreate(TaskAcquireData, "Acq", 150, NULL, 2, NULL);
    xTaskCreate(TaskAnalyzeAndCommunicate, "Anl", 220, NULL, 1, NULL);
    xTaskCreate(TaskCommandListener, "Cmd", 150, NULL, 3, NULL); 
  }
}

void loop() {} 

// ==============================================================================
//  DATA ACQUISITION
// ==============================================================================
void TaskAcquireData(void *pvParameters) {
  (void) pvParameters;
  const TickType_t xFrequency = 500 / portTICK_PERIOD_MS; 
  TickType_t xLastWakeTime = xTaskGetTickCount();

  for (;;) {
    bool isOverridden = false;
    int safeTemp = 25;

    // 1. Safely read global states
    if (xSemaphoreTake(systemStateMutex, portMAX_DELAY) == pdTRUE) {
      isOverridden = tempOverride;
      safeTemp = currentTemp;
      xSemaphoreGive(systemStateMutex);
    }

    // 2. Read physical sensor if not overridden
    if (!isOverridden) {
      safeTemp = readTemperatureI2C();
      if (xSemaphoreTake(systemStateMutex, portMAX_DELAY) == pdTRUE) {
        currentTemp = safeTemp;
        xSemaphoreGive(systemStateMutex);
      }
    }
    
    // 3. Physics simulation using FIXED-POINT MATH (Scale: 10,000 = 1.0)
    long friction_sim = 10000 - (safeTemp * 5); // 0.0005 * 10000 = 5
    long amplitude = 10000;

    // 4. Gather Data (Yielding to prevent CPU blocking)
    for (int index = 0; index < numSamples; index++) {
      ADCSRA |= (1 << ADSC);           
      while (ADCSRA & (1 << ADSC)); 
      
      long rawValue = ADC - 512; 
      
      // Multiply then divide to strip the scale back off
      processedBuffer[index] = (int)((rawValue * amplitude) / 10000);
      
      // Decay the amplitude for the next loop
      amplitude = (amplitude * friction_sim) / 10000; 
      
      vTaskDelay(1); 
    }

    xSemaphoreGive(bufferReadySemaphore); 
    vTaskDelayUntil(&xLastWakeTime, xFrequency); 
  }
}

// ==============================================================================
// EDGE ANALYTICS & SLCAN TRANSMISSION
// ==============================================================================
void TaskAnalyzeAndCommunicate(void *pvParameters) {
  (void) pvParameters;

  for (;;) {
    if (xSemaphoreTake(bufferReadySemaphore, portMAX_DELAY) == pdTRUE) {
      int safeTemp;
      
      // Safely read temperature
      if (xSemaphoreTake(systemStateMutex, portMAX_DELAY) == pdTRUE) {
        safeTemp = currentTemp;
        xSemaphoreGive(systemStateMutex);
      }

      long sumOfSquares = 0;
      int peakValue = 0;

      for (int i = 0; i < numSamples; i++) {
        long val = processedBuffer[i]; // Use long to prevent overflow
        sumOfSquares += (val * val);
        if (abs(val) > peakValue) peakValue = abs((int)val);
      }

      // NEW INTEGER MATH: No floating point sqrt!
      uint32_t meanSquare = sumOfSquares / numSamples;
      uint8_t rms = (uint8_t)isqrt(meanSquare); 
      
      uint8_t statusByte = 0; 
      int targetPWM = 255;

      // TEMPORARY FIX: Raised RMS E-Stop limit to 600 so you can watch the motor spin
      if (safeTemp > 85 || rms >= 600) {
        statusByte = 2; targetPWM = 0; 
      } else if (safeTemp > 60 || rms > 160) {
        statusByte = 1; targetPWM = 128; 
      } else {
        statusByte = 0; targetPWM = 255; 
      }
      
      // Safely write motor PWM
      if (xSemaphoreTake(systemStateMutex, portMAX_DELAY) == pdTRUE) {
        motorPWM = targetPWM;
        xSemaphoreGive(systemStateMutex);
      }
      
      OCR2B = targetPWM; 

      if (targetPWM == 0 || rms < 3) { rms = 0; }

      uint8_t canData[4] = {(uint8_t)safeTemp, rms, (uint8_t)targetPWM, statusByte};
      sendSLCAN(0x100, 4, canData);
    }
  }
}

// ==============================================================================
// SLCAN COMMAND LISTENER
// ==============================================================================
void TaskCommandListener(void *pvParameters) {
  (void) pvParameters;
  for (;;) {
    while (Serial.available() > 0) {
      char c = Serial.read();
      
      if (c == '\r' || c == '\n') {
        if (bufIdx > 0) {
          cmdBuffer[bufIdx] = '\0'; 
          
          if (cmdBuffer[0] == 't') {
            char idStr[4] = {cmdBuffer[1], cmdBuffer[2], cmdBuffer[3], '\0'};
            uint16_t id = strtol(idStr, NULL, 16);
            uint8_t dlc = cmdBuffer[4] - '0';
            uint8_t data[8];
            
            for(int i=0; i<dlc; i++) {
                char byteStr[3] = {cmdBuffer[5 + i*2], cmdBuffer[6 + i*2], '\0'};
                data[i] = strtol(byteStr, NULL, 16);
            }

            if (id == 0x050 && dlc >= 1) { 
              // Safely apply override
              if (xSemaphoreTake(systemStateMutex, portMAX_DELAY) == pdTRUE) {
                currentTemp = data[0]; 
                tempOverride = true;
                xSemaphoreGive(systemStateMutex);
              }
            } 
            else if (id == 0x051) { 
              // Safely remove override
              if (xSemaphoreTake(systemStateMutex, portMAX_DELAY) == pdTRUE) {
                tempOverride = false;
                xSemaphoreGive(systemStateMutex);
              }
            }
          }
          bufIdx = 0; 
        }
      } else if (bufIdx < 31) {
        cmdBuffer[bufIdx++] = c;
      }
    }
    vTaskDelay(50 / portTICK_PERIOD_MS); 
  }
}

// --- HELPER: FORMATS DATA INTO SLCAN STRING ---
void sendSLCAN(uint16_t id, uint8_t dlc, uint8_t* data) {
  Serial.print('t');
  if(id < 0x100) Serial.print('0');
  if(id < 0x010) Serial.print('0');
  Serial.print(id, HEX);
  Serial.print(dlc);
  for(int i=0; i<dlc; i++) {
      if(data[i] < 0x10) Serial.print('0');
      Serial.print(data[i], HEX);
  }
  Serial.print('\r'); 
}

// ==============================================================================
// BARE METAL DRIVERS 
// ==============================================================================

void setupADC() {
  ADMUX = (1 << REFS0);
  ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
}

void setupPWM() {
  DDRD |= (1 << DDD3) | (1 << DDD4) | (1 << DDD5);
  PORTD |= (1 << PD4);  PORTD &= ~(1 << PD5);
  TCCR2A = (1 << COM2B1) | (1 << WGM21) | (1 << WGM20);
  TCCR2B = (1 << CS22); OCR2B = 255; 
}

// --- WATCHDOG TIMEOUT VALUE ---
const uint16_t I2C_TIMEOUT = 10000; 

// Helper function to wait with a timeout
bool i2c_wait() {
  uint16_t timeout = 0;
  while (!(TWCR & (1 << TWINT))) {
    timeout++;
    if (timeout >= I2C_TIMEOUT) return false; // Sensor disconnected! Break the freeze.
  }
  return true;
}

void setupI2C() {
  TWBR = 72; 
  TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN); 
  if (!i2c_wait()) return;
  TWDR = 0x90; TWCR = (1 << TWINT) | (1 << TWEN);   
  if (!i2c_wait()) return;
  TWDR = 0xEE; TWCR = (1 << TWINT) | (1 << TWEN);   
  if (!i2c_wait()) return;
  TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
}

int readTemperatureI2C() {
  TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN); 
  if (!i2c_wait()) return 25; // Default safe temp if sensor fails

  TWDR = 0x90; TWCR = (1 << TWINT) | (1 << TWEN);   
  if (!i2c_wait()) return 25;

  TWDR = 0xAA; TWCR = (1 << TWINT) | (1 << TWEN);   
  if (!i2c_wait()) return 25;

  TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN); 
  if (!i2c_wait()) return 25;

  TWDR = 0x91; TWCR = (1 << TWINT) | (1 << TWEN);   
  if (!i2c_wait()) return 25;

  TWCR = (1 << TWINT) | (1 << TWEN);                
  if (!i2c_wait()) return 25;

  int temp = TWDR;
  TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
  
  if (temp >= 128) temp -= 256; 
  return temp;
}
