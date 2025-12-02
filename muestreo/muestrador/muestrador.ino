const int sensorPin = A0; 
const long samplePeriod_us = 20000; 
// Frecuencia de muestreo: 50Hz
//Período de muestreo 20000 microsegundos

unsigned long lastSampleTime = 0; 
// Tiempo en el que se tomó la última muestra

void setup() {
  Serial.begin(115200); 
  Serial.println("Iniciando muestreo del sensor de lluvia...");
  
  delay(100);
}

void loop() {
  unsigned long currentTime = micros(); 

  // Comprueba si ha pasado el tiempo de muestreo requerido (20 ms)
  if (currentTime - lastSampleTime >= samplePeriod_us) {
    
    //Lectura y transmisión del dato
    int rawValue = analogRead(sensorPin); 
    Serial.println(rawValue); 
    
    // Actualiza el tiempo de la última muestra para el próximo ciclo
    lastSampleTime = currentTime; 
  }
}