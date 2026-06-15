#include "camera_stream.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "pins.h"

#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "esp_log.h"

#define UDP_SERVER_PORT 5000
#define UDP_SERVER_BUF 64
#define CHUNK_SIZE 1200

static bool isInit = false;
static const char* TAG = "CAMERA";

static struct sockaddr_in video_client;
static bool video_client_valid = false;
static char rx_buf[UDP_SERVER_BUF];
static char tx_buf[UDP_SERVER_BUF];
static int global_sock = -1;
static int udp_sock = -1;
static uint16_t frame_counter = 0;

typedef struct{
    uint16_t frame_id;
    uint16_t chunk_id;
    uint16_t chunk_count;
} __attribute__((packed)) video_hdr_t;


bool cameraTest(void){
    return isInit;
};

void cameraStreamInit(void)
{
    if(isInit){
        return;
    }
    ESP_LOGI(TAG, "PSRAM initialized: %s",
         esp_psram_is_initialized() ? "YES" : "NO");

    ESP_LOGI(TAG, "PSRAM size: %u",
            esp_psram_get_size());

    ESP_LOGI(TAG, "Free SPIRAM: %u",
            heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    ESP_LOGI(TAG, "Largest SPIRAM block: %u",
            heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));

    // Initializing camera

    // Camera configuration for XIAO ESP32S3 Sense with OV3660
    static camera_config_t config = {
        // Pin configuration
        .pin_pwdn     = XIAO_CAM_PIN_PWDN,
        .pin_reset    = XIAO_CAM_PIN_RESET,
        .pin_xclk     = XIAO_CAM_PIN_XCLK,
        .pin_sccb_sda = XIAO_CAM_PIN_SIOD,
        .pin_sccb_scl = XIAO_CAM_PIN_SIOC,
        
        .pin_d7       = XIAO_CAM_PIN_D7,
        .pin_d6       = XIAO_CAM_PIN_D6,
        .pin_d5       = XIAO_CAM_PIN_D5,
        .pin_d4       = XIAO_CAM_PIN_D4,
        .pin_d3       = XIAO_CAM_PIN_D3,
        .pin_d2       = XIAO_CAM_PIN_D2,
        .pin_d1       = XIAO_CAM_PIN_D1,
        .pin_d0       = XIAO_CAM_PIN_D0,
        .pin_vsync    = XIAO_CAM_PIN_VSYNC,
        .pin_href     = XIAO_CAM_PIN_HREF,
        .pin_pclk     = XIAO_CAM_PIN_PCLK,
        
        // XCLK settings - OV3660 supports 6-27MHz
        //.xclk_freq_hz = 20000000,
        .xclk_freq_hz = 8000000,
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        
        // Image settings - OV3660 specific
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size   = FRAMESIZE_QVGA,    // Start with QQVGA (160x120) for testing
        .jpeg_quality = 10,                 // 0-63 lower = higher quality
        .fb_count = 3,  // MUST be 2 to prevent FB-OVF errors
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
        
        // OV3660 can do up to QXGA (2048x1536)
        // But start small for testing!
        //FRAMESIZE_QQVGA
    };
    esp_err_t err = esp_camera_init(&config);
    //sensor_t *s = esp_camera_sensor_get();
    //printf("PID: 0x%04x\n", s->id.PID);
    //printf("VER: 0x%04x\n", s->id.VER);

    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        isInit = false;
        return;
    }

    /*
    xTaskCreatePinnedToCore(
        cameraTask,
        "camera",
        8192,
        NULL,
        5,
        NULL,
        1
    );
    ESP_LOGI(TAG, "Camera task started");
    */

    /*
    */
    // Initializing stream server
    ESP_LOGI(TAG, "chk-1");
    xTaskCreatePinnedToCore(
        serverTaskUDP,
        "server",
        4096,
        NULL,
        5,
        NULL,
        1
    );
    ESP_LOGI(TAG, "Server manager started");
    xTaskCreatePinnedToCore(
        streamTaskUDP,
        "stream",
        8192,
        NULL,
        5,
        NULL,
        1
    );
    ESP_LOGI(TAG, "Stream task started");

    isInit = true;
}

void cameraTask(void *arg)
{
    while (1)
    {
        ESP_LOGI(TAG, "cameraTask alive");
        camera_fb_t *fb = esp_camera_fb_get();

        ESP_LOGI(TAG,
        "FB status: len=%u buf=%p",
        fb ? fb->len : 0,
        fb ? fb->buf : NULL);

        if (fb)
        {   
            esp_camera_fb_return(fb);
        }

        vTaskDelay(1);
    }
}

void streamTaskTCP(void *arg){
    while(1){
        //ESP_LOGI(TAG, "streamTask alive");
        if(global_sock < 0){
            vTaskDelay(1);
            continue;
        }
        camera_fb_t *fb = esp_camera_fb_get();
        if(!fb){
            vTaskDelay(1);
            continue;
        }

        if (fb->len == 0 || fb->buf == NULL){
            esp_camera_fb_return(fb);
            continue;
        }
        
        // send size first
        uint32_t len = fb->len;
        send(global_sock, &len, sizeof(len), 0);

        // send JPEG
        int sent = send(global_sock, fb->buf, fb->len, 0);
        esp_camera_fb_return(fb);

        if(sent < 0){
            global_sock = -1;
        }
        vTaskDelay(1);
    }
}

void serverTaskTCP(void *arg){
    char *TAG = "TCP_SERVER";

    int listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if(listen_sock < 0){
        ESP_LOGE(TAG, "Unable to create socket");
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_port = htons(3333);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    int err = bind(listen_sock, (struct sockaddr *)&addr, sizeof(addr));
    if(err != 0){
        ESP_LOGE(TAG, "Bind failed");
        close(listen_sock);
        vTaskDelete(NULL);
        return;
    }

    if(listen(listen_sock, 1) != 0){
        ESP_LOGE(TAG, "Listen failed");
        close(listen_sock);
        vTaskDelete(NULL);
        return;
    }

    //ESP_LOGI("NET", "IP: " IPSTR, IP2STR(&ip_info.ip));
    ESP_LOGI(TAG, "Server listening on port 3333");

    while (1){
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        int sock = accept(listen_sock, (struct sockaddr *)&client_addr, &addr_len);

        if(sock < 0){
            ESP_LOGI(TAG, "Accept failed!");
            continue;
        }

        ESP_LOGI(TAG, "Client connected!");
        global_sock = sock;
        ESP_LOGI(TAG, "New client sock: %d", global_sock);
    }
}

void streamTaskUDP(void* arg){
    while(1){
        if(!video_client_valid){
            vTaskDelay(1);
            continue;
        }
        camera_fb_t *fb = esp_camera_fb_get();
        if(!fb){
            vTaskDelay(1);
            continue;
        }

        if (fb->len == 0 || fb->buf == NULL){
            esp_camera_fb_return(fb);
            vTaskDelay(1);
            continue;
        }

        /*
        ESP_LOGI(TAG, "Sending frame from %d len=%u to client %s:%d", 
        udp_sock,
        fb->len, 
        inet_ntoa(video_client.sin_addr), 
        ntohs(video_client.sin_p`ort));
        */

        uint16_t frame_id = frame_counter++;
        int total_chunks = (fb->len + CHUNK_SIZE - 1) / CHUNK_SIZE;
        for(int i = 0; i < total_chunks; i++){
            video_hdr_t hdr;
            hdr.frame_id = frame_id;
            hdr.chunk_id = i;
            hdr.chunk_count = total_chunks;

            int offset = i * CHUNK_SIZE;
            int size = fb->len - offset;
            if(size > CHUNK_SIZE) size = CHUNK_SIZE;
            
            uint8_t packet[sizeof(video_hdr_t) + CHUNK_SIZE];
            memcpy(packet, &hdr, sizeof(hdr));
            memcpy(packet + sizeof(hdr), fb->buf + offset, size);

            sendto(
                udp_sock,
                packet,
                sizeof(hdr) + size,
                0,
                (struct sockaddr *)&video_client,
                sizeof(video_client)
            );
        }        

        //ESP_LOGI(TAG, "sendto ret=%d errno=%d", ret, errno);
        
        esp_camera_fb_return(fb);
        vTaskDelay(1);
    }
}

void serverTaskUDP(void* arg){
    char *TAG = "UDP_SERVER";

    // Initialize the socket
    udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if(udp_sock < 0){
        ESP_LOGE(TAG, "Failed to create socket");
        return;
    }
    ESP_LOGI(TAG, "UDP socket created successfully: %d", udp_sock);

    // Binding the socket to the port
    struct sockaddr_in listen_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(5000),
        .sin_addr.s_addr = htonl(INADDR_ANY)
    };

    int err = bind(
        udp_sock,
        (struct sockaddr *)&listen_addr,
        sizeof(listen_addr)
    );

    if(err < 0){
        ESP_LOGE(TAG, "bind failed");
        close(udp_sock);
        return;
    }
    ESP_LOGI(TAG, "UDP binded successfully");

    // Listen for incoming package
    struct sockaddr_in sender;
    socklen_t sender_len = sizeof(sender);
    while(1){
        int len = recvfrom(
            udp_sock,
            rx_buf,
            sizeof(rx_buf) - 1,
            0,
            (struct sockaddr *)&sender,
            &sender_len
        );

        if(len > 0){
            rx_buf[len] = '\0';
            ESP_LOGI(TAG, "Client %s:%d", inet_ntoa(sender.sin_addr), ntohs(sender.sin_port));

            char *rx_ptr = &rx_buf[0];
            if(strcmp(rx_ptr, "CLOSE") == 0){
                if(video_client_valid && sender.sin_addr.s_addr == video_client.sin_addr.s_addr &&
                sender.sin_port == video_client.sin_port) video_client_valid = false;
            }else if(strcmp(rx_ptr, "OPEN") == 0){
                if(!video_client_valid){
                    video_client = sender;
                    video_client_valid = true;
                }
            }
        }
    }
    return;
}
