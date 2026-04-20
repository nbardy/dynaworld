# Architecture Diagram: Dyna World v1 Beta

This diagram illustrates the data flow for training the Dyna World v1 Beta architecture on long-form video. It highlights the single-step diffusion preimage, the Causal Splat Decoder, and the ChopGrad truncation barrier.

```mermaid
graph TD
    %% Inputs
    V_IN[Input Video Frames] --> VAE_ENC[Video VAE Encoder]
    VAE_ENC --> L_IN[Noisy Video Latents]
    
    %% Backbone (Single-Step Preimage)
    subgraph "Single-Step Video Diffusion Backbone"
        L_IN --> UNET[Video U-Net / DiT\nLocked t=T]
        UNET -- Extract --> F_T[Spatiotemporal Features\n'Preimage' F_t]
    end

    %% Causal Splat Decoder
    subgraph "Causal Splat Decoder (Recurrent)"
        S_PREV[(Previous Splat State S_t-1)]
        
        F_T --> HEAD[Splat Head / MLP]
        S_PREV --> HEAD
        HEAD --> S_CURR[(Current Splat State S_t)]
        
        %% Recurrence
        S_CURR -.->|Next Frame| S_PREV
    end

    %% Render & Loss
    subgraph "Physics & Pixel Space"
        S_CURR --> RAST[Splat Rasterizer]
        RAST --> IMG_OUT[Rendered RGB Frame t]
        
        GT_FRAME[Ground Truth Video Frame t] --> LOSS{Videometric Loss\nL1 + SSIM}
        IMG_OUT --> LOSS
    end

    %% Backpropagation & ChopGrad
    LOSS -.->|Exact Gradient| RAST
    RAST -.->|Exact Gradient| S_CURR
    S_CURR -.->|Gradient to Features| F_T
    S_CURR -.->|Gradient Through Time| S_PREV
    
    %% The ChopGrad Barrier
    CHOP[fa:fa-cut ChopGrad Truncation Barrier]
    S_PREV -.->|K Frames Max| CHOP
    
    style CHOP fill:#ff4444,stroke:#333,stroke-width:2px,color:#fff
    style RAST fill:#44ff44,stroke:#333,stroke-width:2px,color:#000
    style LOSS fill:#ffff44,stroke:#333,stroke-width:2px,color:#000
```

### Key Data Paths

1. **Forward Pass (Green/Solid Lines):** 
   - A video chunk is passed through the backbone for *one single step*.
   - The features $F_t$ are fed into the Causal Splat Decoder alongside the previous frame's splat state $S_{t-1}$.
   - The output $S_t$ is rasterized into an image.
   
2. **Backward Pass (Dashed Lines):**
   - The pixel loss flows backward perfectly through the rasterizer into $S_t$.
   - The gradient splits: part of it updates the UNet features $F_t$, and part of it flows back in time to $S_{t-1}$ to update the recurrent weights.
   - **The Red Barrier:** ChopGrad intervenes. After the gradient flows backward through $K$ splat states (e.g., 4 frames), PyTorch `.detach()` is called. The gradient stops. The UNet computation graph for $t-5$ is completely freed from VRAM.