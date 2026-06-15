idf_component_register(SRCS "camera_stream.c"
                       INCLUDE_DIRS "." "include"
                       REQUIRES esp32-camera esp_psram esp_http_server
)