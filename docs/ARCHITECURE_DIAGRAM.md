```mermaid
graph TD
    subgraph Inputs ["Data Acquisition"]
        TV[Top-View Camera\nOrthographic Projection]
        FPV[First-Person View Camera\nPerspective Projection]
    end

    subgraph FeatureExtraction ["Feature Extraction - per view independently"]
        DINO_TV[DINOv2-ViT-B/14 frozen + LoRA rank=4\nimg_size=448, 1024 patches x 768d]
        DINO_FPV[DINOv2-ViT-B/14 frozen + LoRA rank=4\nimg_size=448, 1024 patches x 768d\nshared weights with top-view]
    end

    subgraph TemporalProcessing ["Temporal Modeling - per view"]
        TEMP_TV[Temporal Transformer\n1 layer, 8 heads\n8 frames -> mean pool -> 768d]
        TEMP_FPV[Temporal Transformer\n1 layer, 8 heads\n8 frames -> mean pool -> 768d]
    end

    subgraph LateFusion ["Late Fusion - early fusion forbidden"]
        CONCAT[Concat\n768d + 768d -> 1536d]
        MLP[Fusion MLP\n1536 -> 512 -> 256\nLayerNorm + GELU + Dropout 0.3]
    end

    subgraph OutputHeads ["Factorised Output Heads - raw logits"]
        ROW_HEAD[Row Head\nFC 256 -> 8\nbinary per row A-H]
        COL_HEAD[Col Head\nFC 256 -> 12\nbinary per col 1-12]
        TYPE_HEAD[Type Head\nFC 256 -> 3\nsingle / full-row / full-col]
    end

    subgraph Decoder ["Type-Conditioned Inference Decoder"]
        DECODE[logits_to_wells_typed\ntype argmax selects decoding strategy\nthreshold=0.4 on row/col logits]
    end

    subgraph Outputs ["System Outputs"]
        WELLS((Well predictions\nrow+col pairs))
        REFUSAL((Confident Refusal\nflag for human review))
    end

    TV --> |8 frames| DINO_TV
    FPV --> |8 frames| DINO_FPV

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

    subgraph SceneClassifier ["FUTURE - Scene Classifier - not implemented"]
        SC_HEAD[Scene Classification Heads\nwell state, tip state, liquid, hands\nmulti-label per P0/P1 object class]
        SC_OUT((Scene Annotations\nstructured JSON per frame))
    end

    MLP -.-> |shared 256d repr| SC_HEAD
    SC_HEAD -.-> SC_OUT

    classDef input fill:#e2e8f0,stroke:#4a5568,stroke-width:2px,color:#1a202c
    classDef model fill:#bee3f8,stroke:#3182ce,stroke-width:2px,color:#2a4365
    classDef fusion fill:#fefcbf,stroke:#d69e2e,stroke-width:2px,color:#744210
    classDef head fill:#e9d8fd,stroke:#805ad5,stroke-width:2px,color:#44337a
    classDef output_pass fill:#c6f6d5,stroke:#38a169,stroke-width:2px,color:#22543d
    classDef output_fail fill:#fed7d7,stroke:#e53e3e,stroke-width:2px,color:#742a2a
    classDef future fill:#f0f4f8,stroke:#a0aec0,stroke-width:1px,color:#718096

    class TV,FPV input
    class DINO_TV,DINO_FPV,TEMP_TV,TEMP_FPV model
    class CONCAT,MLP fusion
    class ROW_HEAD,COL_HEAD,TYPE_HEAD,DECODE head
    class WELLS output_pass
    class REFUSAL output_fail
    class SC_HEAD,SC_OUT future
```

**Legend**
- Solid coloured borders — implemented and running
- Light grey (`future` class) — proposed, not yet in codebase

> **Scene Classifier** full spec: [FEATURE_SCENE_CLASSIFICATION.md](FEATURE_SCENE_CLASSIFICATION.md)
> **Audio/acoustic modality** is deferred (Architecture D) — not shown.
