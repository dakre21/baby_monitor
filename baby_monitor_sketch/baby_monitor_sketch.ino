#include <LiquidCrystal.h>

int cnt = 0;
int led = 13;
int temp_a0 = A0;
int photor_a1 = A1;
float V0 = 5.0;    // 5V supplied
float R1 = 10000;  // 10kOhm on board
float c1 = 0.001129148, c2 = 0.000234125,
      c3 = 0.0000000876741;  // steinhart-hart coeficients for thermistor

byte heart[8] = {0b00000, 0b01010, 0b11111, 0b11111,
                 0b11111, 0b01110, 0b00100, 0b00000};

byte smiley[8] = {0b00000, 0b00000, 0b01010, 0b00000,
                  0b00000, 0b10001, 0b01110, 0b00000};

LiquidCrystal lcd(1, 2, 4, 5, 6, 7);

void setup() {
  pinMode(led, OUTPUT);

  lcd.begin(16, 2);

  lcd.createChar(1, heart);
  lcd.createChar(2, smiley);

  lcd.setCursor(0, 0);
  lcd.print("Beanie Baby!");

  lcd.setCursor(15, 1);
  lcd.write(1);

  Serial.begin(9600);
}

void loop() {
  // Turn LED on to show program is executing
  digitalWrite(led, HIGH);

  // Read temperatue value out into deg C
  int temp_voltage = analogRead(temp_a0);
  float R = R1 * (1023.0 / (float)temp_voltage - 1.0);
  float temp =
      (1.0 / (c1 + c2 * log(R) + c3 * log(R) * log(R) * log(R))) - 273.15;

  // Read photo resistor value as percentage of low to high brightness
  int photor_voltage = analogRead(photor_a1);
  float photor_percent = (float)photor_voltage / V0;

  Serial.print(temp);
  Serial.print(',');
  Serial.println(photor_percent);

  delay(1000);
}
