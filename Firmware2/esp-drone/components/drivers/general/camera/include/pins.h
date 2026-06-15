/*
 * XIAO ESP32S3 Sense Pin Definitions
 * Include: #include "boards/xiao_esp32s3_sense/pins.h"
 */

#ifndef XIAO_ESP32S3_SENSE_PINS_H
#define XIAO_ESP32S3_SENSE_PINS_H

// ============================================================================
// RGB LED
// ============================================================================
#define XIAO_LED_RGB_GPIO           21
#define LED_GPIO                    21    // For generic examples
#define BLINK_GPIO                  21    // For blink example

// ============================================================================
// Digital I/O (D0-D10)
// ============================================================================
#define XIAO_D0                     1
#define XIAO_D1                     2
#define XIAO_D2                     3
#define XIAO_D3                     4
#define XIAO_D4                     5     // I2C SDA
#define XIAO_D5                     6     // I2C SCL
#define XIAO_D6                     43
#define XIAO_D7                     44
#define XIAO_D8                     7     // SPI SCK
#define XIAO_D9                     8     // SPI MOSI
#define XIAO_D10                    9     // SPI MISO

// ============================================================================
// Analog Pins (A0-A2)
// ============================================================================
#define XIAO_A0                     1
#define XIAO_A1                     2
#define XIAO_A2                     3

// ============================================================================
// I2C Default
// ============================================================================
#define XIAO_I2C_SDA                5
#define XIAO_I2C_SCL                6
#define I2C_MASTER_SDA_IO           5     // For generic I2C examples
#define I2C_MASTER_SCL_IO           6

// ============================================================================
// SPI Default
// ============================================================================
#define XIAO_SPI_MISO               9
#define XIAO_SPI_MOSI               8
#define XIAO_SPI_SCK                7

// ============================================================================
// Camera OV2640
// ============================================================================
#define XIAO_CAM_PIN_PWDN           -1
#define XIAO_CAM_PIN_RESET          -1
#define XIAO_CAM_PIN_XCLK           10
#define XIAO_CAM_PIN_SIOD           40
#define XIAO_CAM_PIN_SIOC           39
#define XIAO_CAM_PIN_D7             48
#define XIAO_CAM_PIN_D6             11
#define XIAO_CAM_PIN_D5             12
#define XIAO_CAM_PIN_D4             14
#define XIAO_CAM_PIN_D3             16
#define XIAO_CAM_PIN_D2             18
#define XIAO_CAM_PIN_D1             17
#define XIAO_CAM_PIN_D0             15
#define XIAO_CAM_PIN_VSYNC          38
#define XIAO_CAM_PIN_HREF           47
#define XIAO_CAM_PIN_PCLK           13


// Compatibility defines for camera examples
#define CAMERA_PIN_PWDN             XIAO_CAM_PIN_PWDN
#define CAMERA_PIN_RESET            XIAO_CAM_PIN_RESET
#define CAMERA_PIN_XCLK             XIAO_CAM_PIN_XCLK
#define CAMERA_PIN_SIOD             XIAO_CAM_PIN_SIOD
#define CAMERA_PIN_SIOC             XIAO_CAM_PIN_SIOC
#define CAMERA_PIN_D7               XIAO_CAM_PIN_D7
#define CAMERA_PIN_D6               XIAO_CAM_PIN_D6
#define CAMERA_PIN_D5               XIAO_CAM_PIN_D5
#define CAMERA_PIN_D4               XIAO_CAM_PIN_D4
#define CAMERA_PIN_D3               XIAO_CAM_PIN_D3
#define CAMERA_PIN_D2               XIAO_CAM_PIN_D2
#define CAMERA_PIN_D1               XIAO_CAM_PIN_D1
#define CAMERA_PIN_D0               XIAO_CAM_PIN_D0
#define CAMERA_PIN_VSYNC            XIAO_CAM_PIN_VSYNC
#define CAMERA_PIN_HREF             XIAO_CAM_PIN_HREF
#define CAMERA_PIN_PCLK             XIAO_CAM_PIN_PCLK

// ============================================================================
// Board Info
// ============================================================================
#define XIAO_BOARD_NAME             "XIAO ESP32S3 Sense"
#define XIAO_FLASH_SIZE_MB          8
#define XIAO_PSRAM_SIZE_MB          8
#define XIAO_HAS_CAMERA             1

#endif // XIAO_ESP32S3_SENSE_PINS_H

