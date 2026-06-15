#ifndef CAMERA_STREAM_H
#define CAMERA_STREAM_H
#pragma once

#include <stdbool.h>

bool cameraTest(void);
void cameraStreamInit(void);
void cameraTask(void *arg);
void streamTaskTCP(void *arg);
void serverTaskTCP(void *arg);
void streamTaskUDP(void *arg);
void serverTaskUDP(void *arg);

#endif