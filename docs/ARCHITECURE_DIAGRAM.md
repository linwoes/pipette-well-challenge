```mermaid
graph TD
    %% Input Layer
    subgraph Inputs ["Data Acquisition (Inputs)"]
        TV[Top-View Camera<br/>Orthographic Projection]
        FPV[First-Person View Camera<br/>Perspective Projection]
    end

    %% Feature Extraction — each view encoded independently in its native frame
    subgraph FeatureExtraction ["Feature Extraction (per view, independent)"]
        DINO_TV[DINOv2-ViT-B/14 frozen<br/>+ LoRA adapters rank=4<br/>img_size=448 → 1024 patches × 768d]
        DINO_FPV[DINOv2-ViT-B/14 frozen<br/>+ LoRA adapters rank=4<br/>img_size=448 → 1024 patches × 768d<br/><i>shared weights with top-view</i>]
    end

    %% Temporal processing — per view
    subgraph TemporalProcessing ["Temporal Modeling (per view)"]
        TEMP_TV[Temporal Transformer<br/>1 layer, 8 heads<br/>N=8 frames → mean pool → 768d]
        TEMP_FPV[Temporal Transformer<br/>1 layer, 8 heads<br/>N=8 frames → mean pool → 768d]
    end

    %% Late Fusion — geometrically mandated
    subgraph LateFusion ["Late Fusion (geometrically mandated — early fusion forbidden)"]
        CONCAT[Concat<br/>768d + 768d → 1536d]
        MLP[Fusion MLP<br/>1536 → 512 → 256<br/>LayerNorm + GELU + Dropout 0.3]
    end

    %% Output Heads
    subgraph OutputHeads ["Factorised Output Heads (raw logits)"]
        ROW_HEAD[Row Head<br/>FC 256 → 8<br/>binary per row A–H]
        COL_HEAD[Col Head<br/>FC 256 → 12<br/>binary per col 1–12]
        TYPE_HEAD[Type Head<br/>FC 256 → 3<br/>single / full-row / full-col]
    end

    %% Inference Decoder
    subgraph Decoder ["Type-Conditioned Inference Decoder"]
        DECODE[logits_to_wells_typed<br/>type argmax selects decoding strategy<br/>threshold=0.4 on row/col logits]
    end

    %% Outputs
    subgraph Outputs ["System Outputs"]
        WELLS((Well predictions<br/>list of row, col pairs))
        REFUSAL((Confident Refusal<br/>flag for human review))
    end

    %% Data Flow
    TV --> |8 frames sampled| DINO_TV
    FPV --> |8 frames sampled| DINO_FPV

    DINO_TV --> TEMP_TV
    DINO_FPV --> TEMP_FPV

    TEMP_TV --> |768d| CONCAT
    TEMP_FPV --> |768d| CONCAT

    CONCAT --> MLP

    MLP --> ROW_HEAD
    MLP --> COL_HEAD
    MLP --> TYPE_HEAD

    ROW_HEAD --> DECODE
    COL_HEAD --> DECODE
    TYPE_HEAD --> DECODE

    DECODE --> WELLS
    DECODE -.-> |low confidence| REFUSAL

    %% ── Future: Scene Classifier (proposed, not implemented) ──────────────
    subgraph SceneClassifier ["⬡ FUTURE: Scene Classifier (not implemented)"]
        SC_HEAD[Scene Classification Heads<br/>well state · tip state · liquid · hands<br/>multi-label per P0/P1 object class]
        SC_OUT((Scene Annotations<br/>structured JSON per frame))
    end

    MLP -.->|shared 256d repr| SC_HEAD
    SC_HEAD -.-> SC_OUT

    %% Styling — implemented pipeline
    classDef input fill:#e2e8f0,stroke:#4a5568,stroke-width:2px,color:#1a202c
    classDef model fill:#bee3f8,stroke:#3182ce,stroke-width:2px,color:#2a4365
    classDef fusion fill:#fefcbf,stroke:#d69e2e,stroke-width:2px,color:#744210
    classDef head fill:#e9d8fd,stroke:#805ad5,stroke-width:2px,color:#44337a
    classDef output_pass fill:#c6f6d5,stroke:#38a169,stroke-width:2px,color:#22543d
    classDef output_fail fill:#fed7d7,stroke:#e53e3e,stroke-width:2px,color:#742a2a

    %% Styling — future / proposed (dashed border, muted grey)
    classDef future fill:#f7fafc,stroke:#a0aec0,stroke-width:2px,stroke-dasharray:6 4,color:#4a5568

    class TV,FPV input
    class DINO_TV,DINO_FPV,TEMP_TV,TEMP_FPV model
    class CONCAT,MLP fusion
    class ROW_HEAD,COL_HEAD,TYPE_HEAD,DECODE head
    class WELLS output_pass
    class REFUSAL output_fail
    class SC_HEAD,SC_OUT future
```

**Legend**
- Solid borders — implemented and running in production training
- Dashed grey borders — proposed future features, not yet in codebase

> **Audio/acoustic modality** is also deferred (Architecture D in ARCHITECTURE.md) — not shown here as it is not yet specced at the diagram level.
> **Scene Classifier** full spec: [`docs/FEATURE_SCENE_CLASSIFICATION.md`](FEATURE_SCENE_CLASSIFICATION.md)
