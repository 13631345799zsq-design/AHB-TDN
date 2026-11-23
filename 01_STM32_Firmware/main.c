#include "stm32f10x.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// I2C Fixed Syntax
#define MCP4728_ADDR   0xC0
#define TCA_CONTROL_ADDR    0xE0
#define TCA_FEEDBACK_ADDR   0xE2
#define ADS1115_ADDR     0x90
#define ADS1115_REG_CONV    0x00
#define ADS1115_REG_CONFIG  0x01
#define HEAT_VOLTAGE_EMA_ALPHA  0.50f
#define HEAT_CURRENT_EMA_ALPHA  0.50f
#define FAN_RPM_EMA_ALPHA       0.20f

// --- Software I2C Macros & Functions Start ---
#define SW_I2C_SCL_H()  GPIO_SetBits(GPIOB, GPIO_Pin_6)
#define SW_I2C_SCL_L()  GPIO_ResetBits(GPIOB, GPIO_Pin_6)
#define SW_I2C_SDA_H()  GPIO_SetBits(GPIOB, GPIO_Pin_7)
#define SW_I2C_SDA_L()  GPIO_ResetBits(GPIOB, GPIO_Pin_7)
#define SW_I2C_SDA_READ() GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_7)

// Short delay for I2C timing
void SW_I2C_Delay(void) {
    volatile uint8_t i = 10; 
    while(i--); 
}

// Generate I2C Start condition
void SW_I2C_Start(void) {
    SW_I2C_SDA_H();
    SW_I2C_SCL_H();
    SW_I2C_Delay();
    SW_I2C_SDA_L();
    SW_I2C_Delay();
    SW_I2C_SCL_L();
}

// Generate I2C Stop condition
void SW_I2C_Stop(void) {
    SW_I2C_SDA_L();
    SW_I2C_SCL_L();
    SW_I2C_Delay();
    SW_I2C_SCL_H();
    SW_I2C_SDA_H();
    SW_I2C_Delay();
}

// Wait for Acknowledgement
uint8_t SW_I2C_WaitAck(void) {
    uint8_t errTime = 0;
    SW_I2C_SDA_H(); SW_I2C_Delay();
    SW_I2C_SCL_H(); SW_I2C_Delay();
    while(SW_I2C_SDA_READ()) {
        errTime++;
        if(errTime > 250) {
            SW_I2C_Stop();
            return 1; // Ack failed
        }
    }
    SW_I2C_SCL_L();
    return 0; // Ack OK
}

// Generate Acknowledgement
void SW_I2C_Ack(void) {
    SW_I2C_SCL_L();
    SW_I2C_SDA_L();
    SW_I2C_Delay();
    SW_I2C_SCL_H();
    SW_I2C_Delay();
    SW_I2C_SCL_L();
}

// Generate Non-Acknowledgement
void SW_I2C_NAck(void) {
    SW_I2C_SCL_L();
    SW_I2C_SDA_H();
    SW_I2C_Delay();
    SW_I2C_SCL_H();
    SW_I2C_Delay();
    SW_I2C_SCL_L();
}

// Send one byte via I2C
void SW_I2C_SendByte(uint8_t byte) {
    uint8_t t;
    SW_I2C_SCL_L();
    for(t = 0; t < 8; t++) {
        if((byte & 0x80) >> 7)
            SW_I2C_SDA_H();
        else
            SW_I2C_SDA_L();
        byte <<= 1;
        SW_I2C_Delay();
        SW_I2C_SCL_H();
        SW_I2C_Delay();
        SW_I2C_SCL_L();
        SW_I2C_Delay();
    }
}

// Read one byte via I2C
uint8_t SW_I2C_ReadByte(uint8_t ack) {
    uint8_t i, receive = 0;
    SW_I2C_SDA_H(); // Release SDA to read
    for(i = 0; i < 8; i++) {
        SW_I2C_SCL_L();
        SW_I2C_Delay();
        SW_I2C_SCL_H();
        receive <<= 1;
        if(SW_I2C_SDA_READ()) receive++;
        SW_I2C_Delay();
    }
    if(!ack) SW_I2C_NAck();
    else SW_I2C_Ack();
    return receive;
}
// --- Software I2C Macros & Functions End ---
volatile uint8_t cmd_ready = 0; 
float target_powers[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
uint16_t MCP_vals[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
float v_Sys = 0.0f; 
float v_D_vals[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float I_S_vals[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
uint8_t v_sys_ema_ready = 0;
uint8_t v_d_ema_ready[12] = {0};
uint8_t i_s_ema_ready[12] = {0};
float p_rom[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
int integral_error[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
uint16_t R_vals[12] = {1, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10};
char cmd_buf[100]; 
char print_buf[400]; 
// ------------------- (fan rpm capturing)
float fan_rpm[2] = {0.0f, 0.0f};
float fan_target_rpm[2] = {0.0f, 0.0f};
float fan_pwm_cmd[2] = {0.0f, 0.0f};
float fan_integral[2] = {0.0f, 0.0f};
uint8_t fan_rpm_ema_ready[2] = {0, 0};
volatile uint16_t last_capture_fan[2] = {0.0f, 0.0f};
volatile uint32_t fan_period[2] = {0.0f, 0.0f};
volatile uint8_t fan_ready[2] = {0, 0};

float EMA_Update(float previous, float sample, float alpha, uint8_t *ready) {
    if (*ready == 0) {
        *ready = 1;
        return sample;
    }
    return previous + alpha * (sample - previous);
}

void Update_System_Voltage(float sample) {
    v_Sys = EMA_Update(v_Sys, sample, HEAT_VOLTAGE_EMA_ALPHA, &v_sys_ema_ready);
}

void Update_Heat_Voltage(uint8_t channel, float sample) {
    if (channel < 12) {
        v_D_vals[channel] = EMA_Update(v_D_vals[channel], sample, HEAT_VOLTAGE_EMA_ALPHA, &v_d_ema_ready[channel]);
    }
}

void Update_Heat_Current(uint8_t channel, float sample) {
    if (channel < 12) {
        I_S_vals[channel] = EMA_Update(I_S_vals[channel], sample, HEAT_CURRENT_EMA_ALPHA, &i_s_ema_ready[channel]);
    }
}

void Update_Fan_RPM(uint8_t fan_idx, float sample) {
    if (fan_idx < 2) {
        fan_rpm[fan_idx] = EMA_Update(fan_rpm[fan_idx], sample, FAN_RPM_EMA_ALPHA, &fan_rpm_ema_ready[fan_idx]);
    }
}

void Delay_ms(uint32_t ms) {
    volatile uint32_t i, j;
    for (i = 0; i < ms; i++) {
        for (j = 0; j < 12000; j++) {
        }
    }
}

void RCC_Configuration(void) {
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB | 
                           RCC_APB2Periph_USART1 | RCC_APB2Periph_AFIO | 
                           RCC_APB2Periph_TIM1, ENABLE);
}

void GPIO_Configuration(void) {
    GPIO_InitTypeDef GPIO_InitStructure;
    // UART
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    // I2C (Changed to Out_OD for bit-banging)
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_6 | GPIO_Pin_7;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD; 
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
    // Default high for I2C bus
    GPIO_SetBits(GPIOB, GPIO_Pin_6 | GPIO_Pin_7);
    // Fan Control
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 | GPIO_Pin_11;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    // Fan Speed
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
}

void USART1_Init(void) {
    USART_InitTypeDef USART_InitStructure;
    USART_InitStructure.USART_BaudRate = 115200;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
    USART_Init(USART1, &USART_InitStructure);
    // ------
    USART_DMACmd(USART1, USART_DMAReq_Rx, ENABLE);
    USART_ITConfig(USART1, USART_IT_RXNE, DISABLE); 
    USART_ITConfig(USART1, USART_IT_IDLE, ENABLE);
    // ------
    USART_Cmd(USART1, ENABLE);
}

// Fan PWM Control (TIM1, 25kHz)
void Fan_PWM_Init(void) {
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_OCInitTypeDef TIM_OCInitStructure;
    // 72MHz / (2879+1) = 25kHz
    TIM_TimeBaseStructure.TIM_Period = 2879; 
    TIM_TimeBaseStructure.TIM_Prescaler = 0;
    TIM_TimeBaseStructure.TIM_ClockDivision = 0;
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);
    TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;
    TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
    TIM_OCInitStructure.TIM_Pulse = 0;
    TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;
    TIM_OC1Init(TIM1, &TIM_OCInitStructure); // Fan 1 (PA8)
    TIM_OC4Init(TIM1, &TIM_OCInitStructure); // Fan 2 (PA11)
    TIM_CtrlPWMOutputs(TIM1, ENABLE);
    TIM_Cmd(TIM1, ENABLE);
}

// DMA Function
void DMA_Configuration(void) {
    DMA_InitTypeDef DMA_InitStructure;
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
    DMA_DeInit(DMA1_Channel5);
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&USART1->DR; 
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)cmd_buf;  
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC; 
    DMA_InitStructure.DMA_BufferSize = 100;    
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;  
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;  
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte; 
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
    DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;  
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(DMA1_Channel5, &DMA_InitStructure);
    DMA_Cmd(DMA1_Channel5, ENABLE); 
}

void Fan_Timers_Init(void) {
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_ICInitTypeDef TIM_ICInitStructure;
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2 | RCC_APB1Periph_TIM3, ENABLE);
    TIM_TimeBaseStructure.TIM_Period = 0xFFFF;  
    TIM_TimeBaseStructure.TIM_Prescaler = 640 - 1; 
    TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);
    TIM_ICInitStructure.TIM_ICPolarity = TIM_ICPolarity_Rising;
    TIM_ICInitStructure.TIM_ICSelection = TIM_ICSelection_DirectTI;
    TIM_ICInitStructure.TIM_ICPrescaler = TIM_ICPSC_DIV1;
    TIM_ICInitStructure.TIM_ICFilter = 0x0F;
    TIM_ICInitStructure.TIM_Channel = TIM_Channel_1;
    TIM_ICInit(TIM2, &TIM_ICInitStructure);
    TIM_ICInitStructure.TIM_Channel = TIM_Channel_2;
    TIM_ICInit(TIM2, &TIM_ICInitStructure);
    TIM_ITConfig(TIM2, TIM_IT_CC1 | TIM_IT_CC2, ENABLE);
    TIM_TimeBaseStructure.TIM_Period = 199; 
    TIM_TimeBaseStructure.TIM_Prescaler = 7199; 
    TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);
    TIM_ClearFlag(TIM3, TIM_FLAG_Update);
    TIM_ITConfig(TIM3, TIM_IT_Update, ENABLE);
    TIM_Cmd(TIM2, ENABLE);
    TIM_Cmd(TIM3, ENABLE);
}

void NVIC_Configuration(void) {
    NVIC_InitTypeDef NVIC_InitStructure;
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    // --- USART1_IRQn ---
    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
    // --- TIM2_IRQn ---
    NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1; 
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
    // --- TIM3_IRQn ---
    NVIC_InitStructure.NVIC_IRQChannel = TIM3_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2; // 优先级低于 TIM2 的捕获
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
}

void TIM2_IRQHandler(void) {
    // --- Fan 1 (PA0 -> CC1) ---
    if (TIM_GetITStatus(TIM2, TIM_IT_CC1) != RESET) {
        uint16_t current_capture = TIM_GetCapture1(TIM2);
        fan_period[0] = (current_capture >= last_capture_fan[0]) ? 
                         (current_capture - last_capture_fan[0]) : 
                    (0xFFFF - last_capture_fan[0] + current_capture + 1);
        last_capture_fan[0] = current_capture;
        fan_ready[0] = 1;
        TIM_ClearITPendingBit(TIM2, TIM_IT_CC1);
    }
    // --- Fan 2 (PA1 -> CC2) ---
    if (TIM_GetITStatus(TIM2, TIM_IT_CC2) != RESET) {
        uint16_t current_capture = TIM_GetCapture2(TIM2);
        fan_period[1] = (current_capture >= last_capture_fan[1]) ? 
                    (current_capture - last_capture_fan[1]) : 
                    (0xFFFF - last_capture_fan[1] + current_capture + 1);
        last_capture_fan[1] = current_capture;
        fan_ready[1] = 1; 
        TIM_ClearITPendingBit(TIM2, TIM_IT_CC2);
    }
}

void Fan_RPM_Control(uint8_t fan_idx) {
    if (fan_target_rpm[fan_idx] >= 1000.0f) { 
        // Feedforward
        float feedforward_pwm = (fan_target_rpm[fan_idx] / 9000.0f) * 100.0f;
        // I-tuning
        float error = ((fan_target_rpm[fan_idx] - fan_rpm[fan_idx]) / 9000.0f) * 100.0f;

        if (error < -1.0f) error = -1.0f; if (error > 1.0f) error = 1.0f;
        float ki = 0.8f;
        if (error < 1.0f && error > -1.0f) ki = 0.6f; 
        if (error < 0.4f && error > -0.4f) ki = 0.4f; 
        if (error < 0.2f && error > -0.2f) ki = 0.2f;

        fan_integral[fan_idx] += ki * error;
        if (fan_integral[fan_idx] > 3.0f) fan_integral[fan_idx] = 3.0f;
        if (fan_integral[fan_idx] < -3.0f) fan_integral[fan_idx] = -3.0f;
        float raw_pwm_cmd = feedforward_pwm + fan_integral[fan_idx];
        if (raw_pwm_cmd < 10.0f) raw_pwm_cmd = 10.0f; if (raw_pwm_cmd > 100.0f) raw_pwm_cmd = 100.0f;
        fan_pwm_cmd[fan_idx] = raw_pwm_cmd;
    } else {
        fan_pwm_cmd[fan_idx] = 0.0f;
        fan_integral[fan_idx] = 0.0f;
    }
}

void TIM3_IRQHandler(void) {
    if (TIM_GetITStatus(TIM3, TIM_IT_Update) != RESET) {
        uint16_t current_timer = TIM_GetCounter(TIM2);
        for(uint8_t i = 0; i < 2; i++) {
            if(fan_ready[i]) {
                if(fan_period[i] > 0) Update_Fan_RPM(i, 3000000.0f / (float)fan_period[i]);
                fan_ready[i] = 0;
            } else {
                uint16_t last_cap = last_capture_fan[i]; 
                uint32_t elapsed = 0;
                if (current_timer >= last_cap) {
                    elapsed = current_timer - last_cap;
                } else {
                    elapsed = 0xFFFF - last_cap + current_timer + 1;
                }
                if (elapsed > 20000) {// 100 ms
                    fan_rpm[i] = 0.0f;
                    fan_rpm_ema_ready[i] = 0;
                }
            }
        }
        Fan_RPM_Control(0);
        Fan_RPM_Control(1);
        TIM_SetCompare1(TIM1, (uint16_t)((fan_pwm_cmd[0] * 2879.0f) / 100.0f));
        TIM_SetCompare4(TIM1, (uint16_t)((fan_pwm_cmd[1] * 2879.0f) / 100.0f));
        TIM_ClearITPendingBit(TIM3, TIM_IT_Update);
    }
}

// Rewritten with Software I2C
void I2C_WriteSingleByte(uint8_t devAddr, uint8_t data) {
    SW_I2C_Start();
    SW_I2C_SendByte(devAddr); // Write Address
    SW_I2C_WaitAck();
    SW_I2C_SendByte(data);
    SW_I2C_WaitAck();
    SW_I2C_Stop();
}

// Rewritten with Software I2C
void MCP4728_Write_Single_Channel(uint8_t channel, uint16_t val) {
    uint8_t cmd;
    switch(channel) {
        case 0: cmd = 0x40; break; // 01000 00 0 
        case 1: cmd = 0x42; break; // 01000 01 0 
        case 2: cmd = 0x44; break; // 01000 10 0 
        case 3: cmd = 0x46; break; // 01000 11 0 
        default: return;
    }
    SW_I2C_Start();
    SW_I2C_SendByte(MCP4728_ADDR);
    SW_I2C_WaitAck();
    SW_I2C_SendByte(cmd); 
    SW_I2C_WaitAck();
    SW_I2C_SendByte((uint8_t)((val >> 8) & 0x0F)); 
    SW_I2C_WaitAck();
    SW_I2C_SendByte((uint8_t)(val & 0xFF));
    SW_I2C_WaitAck();
    SW_I2C_Stop();
    for(volatile int i = 0; i < 300; i++); 
}

void USART1_IRQHandler(void) {
    if(USART_GetITStatus(USART1, USART_IT_IDLE) != RESET) {
        (void)USART1->SR;
        (void)USART1->DR;
        DMA_Cmd(DMA1_Channel5, DISABLE);
        uint16_t rx_len = 100 - DMA_GetCurrDataCounter(DMA1_Channel5);
        if(rx_len < 100) {
            if(cmd_buf[rx_len - 1] == '\n' || cmd_buf[rx_len - 1] == '\r') {
                cmd_buf[rx_len - 1] = '\0';
            } else {
                cmd_buf[rx_len] = '\0';
            }
        } else {
            cmd_buf[99] = '\0'; 
        }
        cmd_ready = 1;
        DMA_SetCurrDataCounter(DMA1_Channel5, 100); 
        DMA_Cmd(DMA1_Channel5, ENABLE); 
    }
}

void UART1_SendString(char* str) {
    while (*str) {
        USART_SendData(USART1, (uint16_t)*str);
        while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
        str++;
    }
}

void SystemClock_Config_HSI(void) {
    RCC_DeInit();
    RCC_HSICmd(ENABLE);
    while(RCC_GetFlagStatus(RCC_FLAG_HSIRDY) == RESET);
    // HSI=8MHz, 8/2 * 16 = 64MHz
    RCC_PLLConfig(RCC_PLLSource_HSI_Div2, RCC_PLLMul_16); 
    RCC_PLLCmd(ENABLE);
    while(RCC_GetFlagStatus(RCC_FLAG_PLLRDY) == RESET);
    RCC_SYSCLKConfig(RCC_SYSCLKSource_PLLCLK);
    while(RCC_GetSYSCLKSource() != 0x08);
}

// Rewritten with Software I2C
void ADS1115_StartConversion(uint8_t channel) {
    uint8_t configHigh = 0xC3 + (channel << 4);
    SW_I2C_Start();
    SW_I2C_SendByte(ADS1115_ADDR); 
    SW_I2C_WaitAck();
    SW_I2C_SendByte(ADS1115_REG_CONFIG); 
    SW_I2C_WaitAck();
    SW_I2C_SendByte(configHigh); 
    SW_I2C_WaitAck();
    SW_I2C_SendByte(0xE3); 
    SW_I2C_WaitAck();
    SW_I2C_Stop();
}

// Rewritten with Software I2C
float ADS1115_ReadResult(void) {
    uint8_t MSB, LSB;
    int16_t val;
    // Set reading pointer
    SW_I2C_Start();
    SW_I2C_SendByte(ADS1115_ADDR); 
    SW_I2C_WaitAck();
    SW_I2C_SendByte(ADS1115_REG_CONV); 
    SW_I2C_WaitAck();
    SW_I2C_Stop();
    // Read data
    SW_I2C_Start();
    SW_I2C_SendByte(ADS1115_ADDR | 0x01); // 0x01 means READ
    SW_I2C_WaitAck();
    MSB = SW_I2C_ReadByte(1); // Send ACK
    LSB = SW_I2C_ReadByte(0); // Send NACK (end reading)
    SW_I2C_Stop();
    val = (MSB << 8) | LSB;
    return (float)val * 0.000125f;
}

// TCA9548A Channel (SDX/SCX) Select
void TCA9548A_Select(uint8_t devAddr, uint8_t channel) {
    I2C_WriteSingleByte(devAddr, (1 << channel));
}

// --- main ---
int main(void) {
    SystemClock_Config_HSI(); 
    RCC_Configuration(); 
    GPIO_Configuration();
    Fan_PWM_Init();
    Fan_Timers_Init();
    DMA_Configuration();
    USART1_Init();
    NVIC_Configuration();
    
    while(1) {
        // negative feedback control
        if (cmd_ready) {
            int chan_num = 0;
            float target_val = 0; float target_val_F1 = 0; float target_val_F2 = 0; 
            // Heat Source Control ----------------------------------
            // Single Command (H1:1.5)
            if (sscanf(cmd_buf, "H%d:%f", &chan_num, &target_val) == 2) {
                    if(chan_num >= 1 && chan_num <= 12) {
                        target_powers[chan_num - 1] = target_val;
                    }
            }
            // Multiple Command (MH:1.5,2.0,3.0)
            else if (sscanf(cmd_buf, "MH:%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", 
                &p_rom[0], &p_rom[1], &p_rom[2], &p_rom[3], &p_rom[4], &p_rom[5], 
                &p_rom[6], &p_rom[7], &p_rom[8], &p_rom[9], &p_rom[10], &p_rom[11]) == 12) {
                target_powers[0] = p_rom[0]; target_powers[1] = p_rom[1]; target_powers[2] = p_rom[2];
                target_powers[3] = p_rom[3]; target_powers[4] = p_rom[4]; target_powers[5] = p_rom[5];
                target_powers[6] = p_rom[6]; target_powers[7] = p_rom[7]; target_powers[8] = p_rom[8];
                target_powers[9] = p_rom[9]; target_powers[10] = p_rom[10]; target_powers[11] = p_rom[11];
            }
            // ------------------------------------------------------
            // Fan Control ----------------------------------
            // Single Command (F1:3000)
            else if (sscanf(cmd_buf, "F%d:%f", &chan_num, &target_val) == 2) {
                if (target_val < 0.0f) target_val = 0.0f;
                if (chan_num == 1) {
                    fan_target_rpm[0] = target_val;
                } else if (chan_num == 2) {
                    fan_target_rpm[1] = target_val;
                }
            }
            // Multiple Command (MF:3000,3500)
            else if (sscanf(cmd_buf, "MF:%f,%f", &target_val_F1, &target_val_F2) == 2) {
                if (target_val_F1 < 0.0f) target_val_F1 = 0.0f;
                if (target_val_F2 < 0.0f) target_val_F2 = 0.0f;
                fan_target_rpm[0] = target_val_F1;
                fan_target_rpm[1] = target_val_F2;
            }
            // Multiple Multiple Command (MF:.....)
            else if (sscanf(cmd_buf, "MHF:%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", 
                &p_rom[0], &p_rom[1], &p_rom[2], &p_rom[3], &p_rom[4], &p_rom[5], 
                &p_rom[6], &p_rom[7], &p_rom[8], &p_rom[9], &p_rom[10], &p_rom[11], 
                &target_val_F1, &target_val_F2) == 14) {
                for(int k=0; k<12; k++) target_powers[k] = p_rom[k];
                fan_target_rpm[0] = target_val_F1;
                fan_target_rpm[1] = target_val_F2;
            }
            cmd_ready = 0;
        }
        // [Reading Part]--------------------------------------------
        // CHannel 0
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 0); ADS1115_StartConversion(0);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 1); ADS1115_StartConversion(0);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); ADS1115_StartConversion(0);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); ADS1115_StartConversion(0);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); ADS1115_StartConversion(0);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); ADS1115_StartConversion(0);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); ADS1115_StartConversion(0);
        Delay_ms(10);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 0); Update_System_Voltage(ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 1); Update_Heat_Voltage(1, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); Update_Heat_Voltage(2, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); Update_Heat_Voltage(4, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); Update_Heat_Voltage(6, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); Update_Heat_Voltage(8, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); Update_Heat_Voltage(10, ADS1115_ReadResult() * 11.0f);
        // CHannel 1
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 0); ADS1115_StartConversion(1); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 1); ADS1115_StartConversion(1); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); ADS1115_StartConversion(1); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); ADS1115_StartConversion(1); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); ADS1115_StartConversion(1); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); ADS1115_StartConversion(1); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); ADS1115_StartConversion(1);
        Delay_ms(10);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 0); Update_Heat_Voltage(0, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 1); Update_Heat_Current(1, ADS1115_ReadResult() / R_vals[1]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); Update_Heat_Current(2, ADS1115_ReadResult() / R_vals[2]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); Update_Heat_Current(4, ADS1115_ReadResult() / R_vals[4]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); Update_Heat_Current(6, ADS1115_ReadResult() / R_vals[6]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); Update_Heat_Current(8, ADS1115_ReadResult() / R_vals[8]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); Update_Heat_Current(10, ADS1115_ReadResult() / R_vals[10]);
        // CHannel 2
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 0); ADS1115_StartConversion(2); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); ADS1115_StartConversion(2); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); ADS1115_StartConversion(2); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); ADS1115_StartConversion(2); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); ADS1115_StartConversion(2); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); ADS1115_StartConversion(2); 
        Delay_ms(10);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 0); Update_Heat_Current(0, ADS1115_ReadResult() / R_vals[0]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); Update_Heat_Voltage(3, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); Update_Heat_Voltage(5, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); Update_Heat_Voltage(7, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); Update_Heat_Voltage(9, ADS1115_ReadResult() * 11.0f);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); Update_Heat_Voltage(11, ADS1115_ReadResult() * 11.0f);
        // CHannel 3
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); ADS1115_StartConversion(3);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); ADS1115_StartConversion(3);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); ADS1115_StartConversion(3); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); ADS1115_StartConversion(3); 
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); ADS1115_StartConversion(3); 
        Delay_ms(10);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 2); Update_Heat_Current(3, ADS1115_ReadResult() / R_vals[3]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 3); Update_Heat_Current(5, ADS1115_ReadResult() / R_vals[5]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 4); Update_Heat_Current(7, ADS1115_ReadResult() / R_vals[7]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 5); Update_Heat_Current(9, ADS1115_ReadResult() / R_vals[9]);
        TCA9548A_Select(TCA_FEEDBACK_ADDR, 6); Update_Heat_Current(11, ADS1115_ReadResult() / R_vals[11]);
        // [Control Part 12]--------------------------------------------
        for (int i = 0; i < 12; i++) {
			float v_load = 0.01f;
            if ((v_Sys - v_D_vals[i]) > 0.0f) {
                v_load = v_Sys - v_D_vals[i];
            }
            // Proportional error Calculation
            int32_t error = (int32_t)(0.2f * (target_powers[i] / v_load - I_S_vals[i]) * R_vals[i] * 1241.0f);
            if (error > 20.0f) error = 20.0f; if (error < -20.0f) error = -20.0f;

            float ki = 1.0f;
            if (error < 20.0f && error > -20.0f) ki = 0.6f; 
            if (error < 10.0f && error > -10.0f) ki = 0.4f; 
            if (error < 5.0f && error > -5.0f) ki = 0.2f;

            // Integral error Calculation
            if (target_powers[i] > 0.01f) {
                integral_error[i] += ki * error;
                if (integral_error[i] > 100.0f) integral_error[i] = 100.0f;
                if (integral_error[i] < -100.0f) integral_error[i] = -100.0f;
            } else {
                integral_error[i] = 0;
            }
            int32_t step = (int32_t)(integral_error[i]);
            int32_t next_dac = 0;
            if ((v_Sys - v_D_vals[i]) > 0.0f) {
                next_dac = (int32_t)(sqrt(target_powers[i] * I_S_vals[i] / v_load) * 1241.0f * R_vals[i]) + step;
            } else {
                next_dac = step;
            }
            if (next_dac > 4095) next_dac = 4095;
            if (next_dac < 0) next_dac = 0;
            MCP_vals[i] = (uint16_t)next_dac;
        }
        // [Execute Part]--------------------------------------------
        for(uint8_t i = 0; i < 12; i++) {
            //---------------------------
            uint8_t j = 0;
            if (i >= 4) {j = 1;}
            if (i >= 8) {j = 2;}
            TCA9548A_Select(TCA_CONTROL_ADDR, j);
            //---------------------------
			Delay_ms(1);
            MCP4728_Write_Single_Channel(i - 4*j, MCP_vals[i]);
        }
        // [Print Part]--------------------------------------------
        char fan_buf[100];
        int fp = 0;
        fp += sprintf(fan_buf + fp,
                    "Target: F1:%.0f, F2:%.0f\r\n",
                    fan_target_rpm[0], fan_target_rpm[1]);
        fp += sprintf(fan_buf + fp,
                    "Command: F1:%.1f, F2:%.1f\r\n",
                    fan_pwm_cmd[0], fan_pwm_cmd[1]);
        fp += sprintf(fan_buf + fp,
                    "Integral: F1:%.1f, F2:%.1f\r\n",
                    fan_integral[0], fan_integral[1]);
        UART1_SendString(fan_buf);

        int p = 0;
        p += sprintf(print_buf + p, "V_Sys:%.2f | F1_RPM:%.0f, F2_RPM:%.0f", v_Sys, fan_rpm[0], fan_rpm[1]);
        for (int i = 0; i < 12; i++) {
            p += sprintf(print_buf + p, " | Ch%d_VD:%.2f,IS:%.2f", 
                            i+1, v_D_vals[i], I_S_vals[i]);
        }
        p += sprintf(print_buf + p, "\r\n");
        UART1_SendString(print_buf);	
    }
}
