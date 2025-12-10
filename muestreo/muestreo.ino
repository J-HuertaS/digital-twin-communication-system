void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));  // Semilla para generar números aleatorios
}

void loop() {
  int valor = random(0, 1024);  // Valor entre 0 y 1023
  Serial.println(valor);        // Un valor por línea, como dijiste

  delay(20); // 50 Hz (1000 ms / 50 = 20 ms)
}

