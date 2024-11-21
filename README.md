# OOD-aware-speaker-diarization-OT



This repository presents a Python-based audio processing pipeline for Out-of-Distribution Aware Speaker Diarization. The pipeline integrates advanced techniques, including Mutual Information, Optimal Transport, and Out-of-Distribution Detection, to achieve robust speaker diarization in challenging multi-speaker environments.

The pipeline begins with Voice Activity Detection (VAD) using WebRTC to isolate speech segments from raw audio. Speaker embeddings are then extracted using pyannote.audio, enriched with temporal context and evaluated for speaker separability using Mutual Information. These embeddings are aligned across temporal variations using Optimal Transport, ensuring consistency for downstream clustering.

To handle unseen or out-of-distribution (OOD) speakers, the pipeline employs a Mahalanobis-distance-based OOD detection mechanism that isolates segments deviating from known speaker distributions. Finally, embeddings are clustered into speaker-specific groups, and time segments are mapped to cluster labels, producing diarization outputs.
