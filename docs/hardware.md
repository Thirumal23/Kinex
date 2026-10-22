# KINEX Hardware Specification & Integration Blueprint

## 1. System Overview
The KINEX hardware sub-system comprises an ESP32 microcontroller acquiring analog flex sensor data and 6-axis IMU telemetry, rendering real-time predictions on an ST7735 SPI TFT, and communicating with the laptop host over USB Serial.

```
                           +------------------------+
                           |   ESP32 DevKit V1      |
                           |                        |
5x Flex Dividers --------->| GPIO 32, 33, 34, 35, 36|
MPU6050 (I2C) ------------>| SDA: 21, SCL: 22       |
ST7735 TFT (SPI) <---------| CS:5, DC:2, RST:4, ... |
                           |                        |
                           | USB Serial (115200)    |
                           +-----------┬------------+
                                       │
                                       ▼ USB Cable
                           +------------------------+
                           | Laptop Host            |
                           | - YOLO26 Vision Model  |
                           | - Feature Normalizer   |
                           | - Multimodal Fusion    |
                           +------------------------+
```

---

## 2. Pin Mapping Table

| Component | Pin Function | ESP32 GPIO | Electrical Specs |
|:----------|:-------------|:-----------|:-----------------|
| **Flex 1 (Thumb)** | Analog Out | **GPIO 33** (ADC1_CH5) | 0 to 3.3V |
| **Flex 2 (Index)** | Analog Out | **GPIO 32** (ADC1_CH4) | 0 to 3.3V |
| **Flex 3 (Middle)**| Analog Out | **GPIO 34** (ADC1_CH6) | 0 to 3.3V (Input Only) |
| **Flex 4 (Ring)**  | Analog Out | **GPIO 35** (ADC1_CH7) | 0 to 3.3V (Input Only) |
| **Flex 5 (Little)**| Analog Out | **GPIO 36 / VP** (ADC1_CH0) | 0 to 3.3V (Input Only) |
| **MPU6050** | I2C Data (SDA)| **GPIO 21** | 3.3V Logic |
| **MPU6050** | I2C Clock (SCL)| **GPIO 22** | 3.3V Logic |
| **ST7735 TFT** | Chip Select (CS) | **GPIO 5** | SPI SS |
| **ST7735 TFT** | Data / Command (DC) | **GPIO 2** | Digital Out |
| **ST7735 TFT** | Reset (RST) | **GPIO 4** | Digital Out |
| **ST7735 TFT** | SPI Clock (SCK) | **GPIO 18** | VSPI SCK |
| **ST7735 TFT** | SPI Data (MOSI) | **GPIO 23** | VSPI MOSI |
| **ST7735 TFT** | Backlight (LED/BL)| **3.3V** | Fixed On |

---

## 3. Flex Sensor Conditioning Circuit

Each flex sensor is 2.2 inches in length. In a resting flat state, nominal resistance is $\approx 25\text{ k}\Omega$; when bent $90^\circ$, resistance increases to $\approx 60\text{ k}\Omega - 100\text{ k}\Omega$.

Voltage divider configuration:
```
3.3V (VCC)
  |
  +-- [ Flex Sensor (R_flex) ]
  |
  +---------> ESP32 ADC Input Pin
  |
  +-- [ 68 kΩ Resistor (R_fixed) ]
  |
GND
```

The output voltage $V_{\text{ADC}}$ is:
$$V_{\text{ADC}} = 3.3\text{V} \times \frac{68\text{ k}\Omega}{R_{\text{flex}} + 68\text{ k}\Omega}$$

- Flat ($R_{\text{flex}} \approx 25\text{ k}\Omega$): $V_{\text{ADC}} \approx 3.3 \times \frac{68}{93} \approx 2.41\text{V}$ (ADC count $\approx 2990$)
- Bent ($R_{\text{flex}} \approx 70\text{ k}\Omega$): $V_{\text{ADC}} \approx 3.3 \times \frac{68}{138} \approx 1.62\text{V}$ (ADC count $\approx 2010$)

Per-user calibration normalizes these raw readings into a range of $[0.0, 1.0]$:
$$\text{bend}_{\text{norm}} = \text{clip}\left(\frac{\text{raw} - \text{val}_{\text{open}}}{\text{val}_{\text{closed}} - \text{val}_{\text{open}}}, 0.0, 1.0\right)$$

---

## 4. MPU6050 Orientation Processing

The MPU6050 is sampled at 50 Hz.
Roll and pitch are estimated using a standard complementary filter combining accelerometer tilt and gyroscope angular velocity:
$$\text{Pitch}_{t} = \alpha \left(\text{Pitch}_{t-1} + \omega_y \cdot \Delta t\right) + (1 - \alpha) \cdot \arctan2\left(a_x, \sqrt{a_y^2 + a_z^2}\right)$$
$$\text{Roll}_{t} = \alpha \left(\text{Roll}_{t-1} + \omega_x \cdot \Delta t\right) + (1 - \alpha) \cdot \arctan2\left(a_y, a_z\right)$$
where $\alpha = 0.98$.

---

## 5. Serial Communication Protocol

### Uplink: ESP32 $\to$ Laptop (50 Hz JSON)
```json
{
  "type": "sensor",
  "thumb": 0.15,
  "index": 0.82,
  "middle": 0.91,
  "ring": 0.86,
  "little": 0.74,
  "roll": 11.2,
  "pitch": 73.5
}
```

### Downlink: Laptop $\to$ ESP32 (On Gesture Recognition)
```json
{
  "type": "prediction",
  "gesture": "THANK_YOU",
  "confidence": 0.93
}
```
