

# Install PyTorch (CPU version)
!pip install torch

# Install Pyannote Audio and Hugging Face transformers for speaker embedding models
!pip install pyannote.audio transformers

# Install WebRTC for VAD
!pip install webrtcvad

# Install Librosa and Soundfile for audio processing
!pip install librosa soundfile

# Install scikit-learn for clustering and silhouette score
!pip install scikit-learn

# Install matplotlib for plotting
!pip install matplotlib

# Install POT (Python Optimal Transport) for alignment with optimal transport
!pip install pot

!pip install scikit-learn

import wave
import webrtcvad
import numpy as np
import librosa
import soundfile as sf
import torch
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import ot  # POT for Optimal Transport
from sklearn.metrics import mutual_info_score
import wave
import webrtcvad
import numpy as np
import librosa
import soundfile as sf
import torch
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, mutual_info_score
from scipy.spatial.distance import mahalanobis
import ot
from scipy.spatial.distance import mahalanobis

def read_wave(path):
    """Reads a .wav file and converts it to PCM 16-bit mono format."""
    wf = wave.open(path, 'rb')
    assert wf.getnchannels() == 1  # Mono channel
    assert wf.getsampwidth() == 2  # 16-bit
    sample_rate = wf.getframerate()
    assert sample_rate in (8000, 16000, 32000, 48000)
    pcm_data = wf.readframes(wf.getnframes())
    wf.close()
    return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 16-bit PCM
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames using VAD."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    voiced_frames = []
    for frame in frames:
        if vad.is_speech(frame, sample_rate):
            voiced_frames.append(frame)
    return voiced_frames

# Load audio file
audio_file = '/content/download-14_085DBcws.wav'
y, sr = librosa.load(audio_file, sr=16000, mono=True)
y_pcm = (y * 32767).astype(np.int16)
y_pcm_bytes = y_pcm.tobytes()

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(2)

# Generate frames from PCM data
frames = frame_generator(30, y_pcm_bytes, sr)
voiced_frames = vad_collector(sr, 30, 300, vad, frames)

# Combine voiced frames
voiced_pcm_bytes = b''.join(voiced_frames)
voiced_pcm_array = np.frombuffer(voiced_pcm_bytes, dtype=np.int16)

# Save the voiced audio
def save_voiced_audio(voiced_pcm_array, sample_rate, output_file):
    """Saves the voiced audio (after VAD filtering) to a .wav file."""
    voiced_float = voiced_pcm_array / 32768.0
    sf.write(output_file, voiced_float, sample_rate)
    print(f"Voiced audio saved to {output_file}")

output_file = 'voiced_output.wav'
save_voiced_audio(voiced_pcm_array, sr, output_file)

# Initialize pre-trained speaker embedding model from pyannote.audio
model = Inference("pyannote/embedding", use_auth_token="")

# Extract embeddings with temporal context
# Calculate mutual information for consecutive embeddings
def calculate_mutual_information(embedding1, embedding2):
    """Calculate mutual information between two embeddings."""
    # Convert embeddings to integers for mutual information calculation (rounding may be required)
    embedding1_int = np.round(embedding1 * 1000).astype(int)
    embedding2_int = np.round(embedding2 * 1000).astype(int)
    return mutual_info_score(embedding1_int, embedding2_int)

# Extract embeddings with temporal context and mutual information
def extract_speaker_embeddings(voiced_pcm_array, sample_rate):
    """Extract speaker embeddings with added temporal context and mutual information."""
    embeddings = []
    window_size = sample_rate  # Use 1-second window

    silence_threshold = 0.01
    start_idx = 0
    prev_embedding = None

    for i in range(len(voiced_pcm_array)):
        if abs(voiced_pcm_array[i]) < silence_threshold:
            if i - start_idx >= window_size:
                segment = voiced_pcm_array[start_idx:i]
                segment_float = (segment / 32768.0).astype(np.float32)
                segment_tensor = torch.tensor(segment_float).unsqueeze(0)
                embedding = model({'waveform': segment_tensor, 'sample_rate': sample_rate}).data.flatten()

                # Add temporal context
                feature_vector = np.hstack([embedding, [i / sample_rate]])  # Append timestamp

                # Calculate mutual information with previous embedding if exists
                if prev_embedding is not None:
                    mi = calculate_mutual_information(prev_embedding, embedding)
                else:
                    mi = 0  # No mutual information for the first segment

                feature_vector = np.hstack([feature_vector, [mi]])  # Add mutual information as feature
                embeddings.append(feature_vector)

                prev_embedding = embedding  # Update previous embedding
                start_idx = i + 1

    # Process remaining segment if exists
    if start_idx < len(voiced_pcm_array):
        segment = voiced_pcm_array[start_idx:]
        segment_float = (segment / 32768.0).astype(np.float32)
        segment_tensor = torch.tensor(segment_float).unsqueeze(0)
        embedding = model({'waveform': segment_tensor, 'sample_rate': sample_rate}).data.flatten()

        # Add temporal context and mutual information
        feature_vector = np.hstack([embedding, [len(voiced_pcm_array) / sample_rate]])
        if prev_embedding is not None:
            mi = calculate_mutual_information(prev_embedding, embedding)
        else:
            mi = 0

        feature_vector = np.hstack([feature_vector, [mi]])
        embeddings.append(feature_vector)

    return np.vstack(embeddings)


# Step 2: Use Optimal Transport to align the embeddings
def apply_optimal_transport(embeddings, sample_rate, time_step=1):
    """Apply Optimal Transport to align embeddings."""
    n_samples = len(embeddings)

    source_weights = np.ones((n_samples,)) / n_samples
    target_weights = np.ones((n_samples,)) / n_samples

    cost_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            time_dist = abs(i - j) * time_step / sample_rate
            feature_dist = np.linalg.norm(embeddings[i] - embeddings[j])
            cost_matrix[i, j] = feature_dist + time_dist

    transport_plan = ot.emd(source_weights, target_weights, cost_matrix)
    aligned_embeddings = np.dot(transport_plan, embeddings)

    return aligned_embeddings

# Estimate the number of speakers
def estimate_num_speakers(embeddings, max_speakers=10):
    """Estimate the number of speakers using silhouette score."""
    best_num_speakers = 2
    best_silhouette_score = -1
    for n_clusters in range(2, max_speakers + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_speakers = n_clusters
    return best_num_speakers

# Perform clustering
def perform_clustering(embeddings, num_speakers):
    """Cluster the embeddings."""
    clustering = AgglomerativeClustering(n_clusters=num_speakers)
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

# Map clusters to time segments
def map_clusters_to_segments(cluster_labels, voiced_pcm_array, sample_rate):
    """Maps cluster labels back to time segments."""
    window_size = sample_rate
    time_segments = []

    start_idx = 0
    for i, label in enumerate(cluster_labels):
        if i == len(cluster_labels) - 1 or cluster_labels[i] != cluster_labels[i + 1]:
            end_idx = (i + 1) * window_size if i < len(cluster_labels) - 1 else len(voiced_pcm_array)
            start_time = start_idx / sample_rate
            end_time = min(end_idx / sample_rate, len(voiced_pcm_array) / sample_rate)
            time_segments.append((label, start_time, end_time))
            start_idx = end_idx

    return time_segments

# Example usage
embeddings = extract_speaker_embeddings(voiced_pcm_array, sr)
aligned_embeddings = apply_optimal_transport(embeddings, sample_rate=sr, time_step=1)
num_speakers = estimate_num_speakers(aligned_embeddings, max_speakers=10)
print(f"Estimated number of speakers: {num_speakers}")
cluster_labels = perform_clustering(aligned_embeddings, num_speakers)
diarization_result = map_clusters_to_segments(cluster_labels, voiced_pcm_array, sr)

for speaker, start, end in diarization_result:
    print(f"Speaker {speaker}: from {start:.2f}s to {end:.2f}s")



# Out-of-Distribution Detection Function with dimensionality check
def detect_ood(embedding, known_embeddings, threshold=2.5):
    """Detects if an embedding is out-of-distribution based on known cluster centers."""
    if len(known_embeddings) < 2:
        # Not enough known embeddings to compute covariance, skip OOD check
        return False  # Treat it as in-distribution if OOD check is not possible

    mean_embedding = np.mean(known_embeddings, axis=0)
    cov_matrix = np.cov(known_embeddings, rowvar=False)

    # Avoid errors with singular matrices by using pseudo-inverse
    try:
        dist = mahalanobis(embedding, mean_embedding, np.linalg.pinv(cov_matrix))
    except np.linalg.LinAlgError:
        # If inversion fails, consider as in-distribution (or set a default behavior)
        dist = 0

    return dist > threshold  # Returns True if OOD is detected

# Extract Speaker Embeddings with Temporal Context and OOD Detection
def extract_speaker_embeddings_with_ood(voiced_pcm_array, sample_rate):
    embeddings = []
    window_size = sample_rate
    start_idx = 0
    known_embeddings = []

    for i in range(0, len(voiced_pcm_array), window_size):
        segment = voiced_pcm_array[i:i + window_size]
        segment_float = (segment / 32768.0).astype(np.float32)
        segment_tensor = torch.tensor(segment_float).unsqueeze(0)
        embedding = model({'waveform': segment_tensor, 'sample_rate': sample_rate}).data.flatten()

        # Check for OOD with known embeddings
        if len(known_embeddings) > 0:
            if detect_ood(embedding, np.vstack(known_embeddings)):
                print(f"OOD embedding detected at segment {i / sample_rate}s.")
                continue  # Skip OOD segments

        known_embeddings.append(embedding)
        embeddings.append(embedding)

    return np.vstack(embeddings)

# Process and align embeddings using Optimal Transport
aligned_embeddings = apply_optimal_transport(extract_speaker_embeddings_with_ood(voiced_pcm_array, sr), sample_rate=sr)
num_speakers = estimate_num_speakers(aligned_embeddings, max_speakers=10)

print(f"Estimated number of speakers (after OOD filtering): {num_speakers}")

# Clustering with OOD-handling labels
cluster_labels = perform_clustering(aligned_embeddings, num_speakers)
diarization_result = map_clusters_to_segments(cluster_labels, voiced_pcm_array, sr)
for speaker, start, end in diarization_result:
    print(f"Speaker {speaker}: from {start:.2f}s to {end:.2f}s")

import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.core import Annotation, Segment

# Function to create an Annotation object for diarization metrics
def create_annotation(segments):
    annotation = Annotation()
    for speaker, start, end in segments:
        annotation[Segment(start, end)] = speaker
    return annotation

# Function to calculate custom diarization metrics
def calculate_diarization_metrics(ground_truth_segments, hypothesis_segments, total_duration):
    # Convert to pyannote.core.Annotation format
    ground_truth_annotation = create_annotation(ground_truth_segments)
    hypothesis_annotation = create_annotation(hypothesis_segments)

    # Initialize standard metrics
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()

    # Compute DER and JER
    der_result = der_metric(ground_truth_annotation, hypothesis_annotation)
    jer_result = jer_metric(ground_truth_annotation, hypothesis_annotation)

    # Extract error components from the DER result
    false_alarm_time = der_metric['false alarm']
    missed_detection_time = der_metric['missed detection']
    speaker_confusion_time = der_metric['confusion']

    # False Alarm Rate (FAR) and Missed Detection Rate (MDR)
    false_alarm_rate = false_alarm_time / total_duration
    missed_detection_rate = missed_detection_time / total_duration

    # Speaker Confusion Error Rate
    speaker_confusion_rate = speaker_confusion_time / total_duration

    # Time-based Confusion Error Rate (tCER) and Time-based Diarization Error Rate (tDER)
    total_ref_speech_time = sum((end - start) for _, start, end in ground_truth_segments)
    total_hyp_speech_time = sum((end - start) for _, start, end in hypothesis_segments)

    tcer = speaker_confusion_time / (total_ref_speech_time + total_hyp_speech_time)
    tder = (false_alarm_time + missed_detection_time + speaker_confusion_time) / total_duration

    # Print results capped at 100%
    print(f"Diarization Error Rate (DER): {min(der_result * 100, 100):.4f}%")
    print(f"Jaccard Error Rate (JER): {min(jer_result * 100, 100):.4f}%")
    print(f"False Alarm Rate (FAR): {min(false_alarm_rate * 100, 100):.4f}%")
    print(f"Missed Detection Rate (MDR): {min(missed_detection_rate * 100, 100):.4f}%")
    print(f"Speaker Confusion Error Rate: {min(speaker_confusion_rate * 100, 100):.4f}%")
    print(f"Time-based Confusion Error Rate (tCER): {min(tcer * 100, 100):.4f}%")
    print(f"Time-based Diarization Error Rate (tDER): {min(tder * 100, 100):.4f}%")

# Example segments (in seconds)
ground_truth_segments = [("Speaker 0", 0.0, 6.0), ("Speaker 1", 7.0, 13.0),("Speaker 2", 13.0, 22.0), ("Speaker 3", 22.0, 27.0),("Speaker 4", 27.0, 37.0)]
hypothesis_segments = [("Speaker 0", 0.0, 6.0), ("Speaker 1", 7.0, 20.0),("Speaker 2", 20.0, 27.0),("Speaker 3", 27.0, 37.0)]
total_duration = 22.18  # Total duration of the audio in seconds

# Calculate diarization metrics
calculate_diarization_metrics(ground_truth_segments, hypothesis_segments, total_duration)
