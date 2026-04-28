```mermaid
graph TD
    subgraph Inputs ["Data Acquisition"]
        TV[Top-View Camera<br/>Orthographic Projection]
        FPV[First-Person View Camera<br/>Perspective Projection]
    end

    subgraph FeatureExtraction ["Feature Extraction - per view independently"]
        DINO_TV[DINOv2-ViT-B/14 frozen + LoRA rank=4<br/>img_size=448, 1024 patches x 768d]
        DINO_FPV[DINOv2-ViT-B/14 frozen + LoRA rank=4<br/>img_size=448, 1024 patches x 768d<br/>shared weights with top-view]
    end

    subgraph TemporalProcessing ["Temporal Modeling - per view"]
        TEMP_TV[Temporal Transformer<br/>1 layer, 8 heads<br/>8 frames -&gt; mean pool -&gt; 768d]
        TEMP_FPV[Temporal Transformer<br/>1 layer, 8 heads<br/>8 frames -&gt; mean pool -&gt; 768d]
    end

    subgraph LateFusion ["Late Fusion - early fusion forbidden"]
        CONCAT[Concat<br/>768d + 768d -&gt; 1536d]
        MLP[Fusion MLP<br/>1536 -&gt; 512 -&gt; 256<br/>LayerNorm + GELU + Dropout 0.3]
    end

    subgraph OutputHeads ["Factorised Output Heads - raw logits"]
        ROW_HEAD[Row Head<br/>FC 256 -&gt; 8<br/>binary per row A-H]
        COL_HEAD[Col Head<br/>FC 256 -&gt; 12<br/>binary per col 1-12]
        TYPE_HEAD[Type Head<br/>FC 256 -&gt; 3<br/>single / full-row / full-col]
    end

    subgraph Decoder ["Type-Conditioned Inference Decoder"]
        DECODE[logits_to_wells_typed<br/>type argmax selects decoding strategy<br/>threshold=0.4 on row/col logits]
    end

    subgraph Outputs ["System Outputs"]
        WELLS((Well predictions<br/>row+col pairs))
        REFUSAL((Confident Refusal<br/>flag for human review))
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
        SC_HEAD[Scene Classification Heads<br/>well state, tip state, liquid, hands<br/>multi-label per P0/P1 object class]
        SC_GATE[Safety Gate<br/>hand detected -&gt; suppress row/col<br/>tip cracked / foam -&gt; flag WARNING]
        SC_OUT((Scene Annotations<br/>structured JSON per frame))
    end

    MLP -.-> |shared 256d repr| SC_HEAD
    SC_HEAD -.-> SC_OUT
    SC_HEAD -.-> |hand in frame| SC_GATE
    SC_GATE -.-> |suppress predictions| REFUSAL

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
    class SC_HEAD,SC_GATE,SC_OUT future
```

**Legend**
- Solid coloured borders — implemented and running
- Light grey (`future` class) — proposed, not yet in codebase
- Dotted edges — future data flows (not yet implemented)

> **Scene Classifier** full spec: [FEATURE_SCENE_CLASSIFICATION.md](FEATURE_SCENE_CLASSIFICATION.md)
> **Audio/acoustic modality** is deferred (Architecture D) — not shown.
