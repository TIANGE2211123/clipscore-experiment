# Cross-Dataset GPU Experiment Plan

```mermaid
flowchart TD
    A["Source discovery"] --> B["Download candidate videos"]
    B --> C["Normalize filenames and metadata"]
    C --> D["Sample 100 videos per dataset"]
    D --> E["Export label-prompt CSV"]
    E --> F["Generate safe / near-crash / crash descriptions"]
    F --> G["Extract first and last frames"]
    G --> H["Build FramePack queue bundle"]
    H --> I["Sync bundle to AutoDL"]
    I --> J["Load queue in FramePack-Studio"]
    J --> K["GPU generation on AutoDL"]
    K --> L["Collect videos and manifests"]

    subgraph Dataset A
        A1["DAIR-V2X / DAIR-V2X proxy sequences"]
    end

    subgraph Dataset B
        B1["Euro NCAP crash videos"]
    end

    A1 --> B
    B1 --> B
```

## Output Targets

- `output/<dataset>/videos/`
- `output/<dataset>/manifest.csv`
- `output/<dataset>/label_prompt.csv`
- `output/<dataset>/classified_descriptions.json`
- `output/<dataset>/classified_descriptions.csv`
- `output/<dataset>/frame_refs/`
- `output/<dataset>/queue_images/`
- `output/<dataset>/queue_seed.json`
- `output/<dataset>/job_manifest.csv`
