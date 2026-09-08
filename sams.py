# SAMS
##############################################################
'''
Parameters of an onset: 
    x = [tau, d, r, pt, gt, *ps, *gs]  
    tau: onset timestamp (sec). 
    d: duration (sec).
    r: audio strech rate; speedup r > 1.
    pt: pitches (Hz) of target melodies, (pitches,).
    gt: gains of target melodies, (gains,).
    [todo]
    *ps: pitches (Hz) of synthesized melodies, (pitches,).
    *gs: gains of synthesized melodies, (gains,). 

    ** Only tau can be adjusted by hill climbing
    For n onsets:
        onsets: [tau_1, tau_2, ..., tau_n]
        durations:  [d_1, d_2, ..., d_n]
        strechs:    [r_1, r_2, ..., r_n]
        f0_pitches: [p_1, p_2, ..., p_n]
        melodies: [[(p11, g11), ...], [(p21, g21), ...]..., [(pn1, gn1), ...]]

Parameters of all melodies:
    sigma: Gaussian deviation, [0, 1]  

Rendering parameters of each melody: 
    q = [d, r, p, sigma, gt, tau]   
'''
##############################################################

import os
import sys
import re
import inspect
import librosa
import psola
import json
import numpy as np
import soundfile as sf
import copy
import bisect
import warnings
import matplotlib.pyplot as plt
import scipy.signal
from time import time
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from argparse import ArgumentParser

def nmltxt(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parsePath(filepath):  
    p = Path(filepath)  
    directory = str(p.parent)
    filename = str(p.stem)
    extname = str(p.suffix)
    return directory, filename, extname

# Converting X to list
def tolist(X):    
    return [[v.tolist() if isinstance(v, np.ndarray) else v for v in x] for x in X]

# Converting list to X
def toX(XL):
    return [[ np.array(v) if isinstance(v, list) else v for v in x ]  for x in XL]

# Loading X from json file
#   path: json file path.
#   ele_path: element audio path.
#   sw: mixing switches.
# return:
#   audio_out: synthesized audio
#   DX: dict of X
def loadX(path, ele_path=None, sw = [True, True, True, True, True]):
    with open(path) as f:
        DX = json.load(f)    
        X = toX(DX['X'])
        DX['X'] = X
        sr = DX['sr']
        sigma=DX['sigma']
        n_samples = DX['samples']         

        if ele_path is None:
            ele_path = DX['element_path']
        audio_ele = librosa.load(ele_path, sr=sr)[0]
        
        audio_out = np.zeros(n_samples)
        
        for x_i in X:
            audio_out = mixx(audio_out, audio_ele, x_i, sr, sigma, sw=sw)         
        return audio_out, DX

# Evaluating the speech quality of each melody
#   audio: input audio
#   sr: sampling rate
#   hop_length: hop length
#   delta: onset threshold
#   min_melody_dur: minimum melody duration (sec), it guarantees that the duration of each splitted audio clip >= min_melody_dur
#   dur_thres: it eliminates the short audio clip with duration <= dur_thres (sec)
#   n_fft: fft size
#   adj_pitch: adjusting the pitch of each audio clip
#   zero_clip: clipping negative values to zero.
# Return: value of each clip (melody), (clips,)
def evaluate(audio_t, audio_out, sr,
             delta=0.05, 
             hop_length=512,
             n_fft=512,
             min_melody_dur=0.75,
             dur_thres=0.125,
             n_max_clip=0,
             pitch_mode='log',
             fmin=librosa.note_to_hz('C1'),
             fmax=librosa.note_to_hz('C7'),
             zero_clip=True,
             adj_pitch=True):
    (onsets_t, 
    f0_pitches_t, 
    melodies_t, 
    durations_t, 
    audio_clips_t,
    clip_slices_t) = audiometa(audio=audio_t, 
                              sr=sr, 
                              hop_length=hop_length, 
                              delta=delta, 
                              min_melody_dur=min_melody_dur,
                              dur_thres=dur_thres,
                              n_max=n_max_clip, 
                              n_fft=n_fft)  
    (onsets_x, 
    f0_pitches_x, 
    melodies_x, 
    durations_x, 
    audio_clips_x,
    clip_slices_x) = audiometa(audio=audio_out,
                              sr=sr, 
                              hop_length=hop_length, 
                              delta=delta, 
                              min_melody_dur=min_melody_dur,
                              dur_thres=dur_thres,
                              n_max=n_max_clip, 
                              n_fft=n_fft)        
    
    e_onset, e_pitch , _, _ = scoredist(ref_onsets=onsets_t, 
                               ref_pitches=f0_pitches_t, 
                               est_onsets=onsets_x, 
                               est_pitches=f0_pitches_x, 
                               pitch_mode=pitch_mode)
    ev = (float(e_onset), float(e_pitch),  )    
    return ev 


def evaluatemeta(meta_t, meta_out, pitch_mode='log'):
    (onsets_t, f0_pitches_t, melodies_t, durations_t, audio_clips_t, clip_slices_t) = meta_t  
    (onsets_x, f0_pitches_x, melodies_x, durations_x, audio_clips_x, clip_slices_x) = meta_out      
    e_onset, e_pitch , _, _ = scoredist(ref_onsets=onsets_t, 
                               ref_pitches=f0_pitches_t, 
                               est_onsets=onsets_x, 
                               est_pitches=f0_pitches_x, 
                               pitch_mode=pitch_mode)
    ev = (float(e_onset), float(e_pitch), )    
    return ev 


# Evaluating the quality of each melody
#   audio: input audio
#   audio_ele: target speech audio
#   sr: sampling rate
#   hop_length: hop length
#   delta: onset threshold
#   min_melody_dur: minimum melody duration (sec), it guarantees that the duration of each splitted audio clip >= min_melody_dur
#   dur_thres: it eliminates the short audio clip with duration <= dur_thres (sec)
#   n_fft: fft size
#   adj_pitch: adjusting the pitch of each audio clip
#   zero_clip: clipping negative values to zero.
# Return: stoi value of each clip (melody), (clips,)
def evalstoi(audio, audio_ele, sr, 
            hop_length=512,
            delta=0.05,
            n_fft=512,
            min_melody_dur = 0.5, 
            dur_thres=0.125, 
            fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C7'),
            adj_pitch=True,
            zero_clip=True
            ): 
    (onsets, 
    f0_pitches, 
    melodies, 
    durations, 
    audio_clips,
    clip_slices) = audiometa(audio, sr, 
                            hop_length=hop_length, 
                            delta=delta, 
                            min_melody_dur=min_melody_dur, 
                            dur_thres=dur_thres,
                            n_fft=n_fft)
    stois = []
    for i, audio_clip_i in enumerate(audio_clips): 
        r = audio_ele.size / audio_clip_i.size
        audio_ele_i = audiostretch(audio_ele, r, n_fft=n_fft)
        #print('audio_clip, audio_ele_i', audio_clip_i.size, audio_ele_i.size)
        
        if adj_pitch:
            audio_ele_i = psola.vocode(audio_ele_i, sample_rate=sr, 
                             target_pitch=[f0_pitches[i],], 
                             fmin=fmin, fmax=fmax)            
        d_stoi = stoi(audio_clip_i, audio_ele_i, sr, extended=False)
        if zero_clip:
            d_stoi = max(0, d_stoi)        
        #print(f'd_stoi:{d_stoi:.4f}')
        stois.append(d_stoi)
    stois = np.array(stois)
    return stois    

# Evaluates temporal and pitch accuracy using global optimal alignment.
#   ref_onsets, ref_pitches: GT timestamps (s) and frequencies (Hz)
#   est_onsets, est_pitches: Predicted timestamps (s) and frequencies (Hz)
#   pitch_mode: 'raw' for Hz, 'log' for musical cents (recommended)
def scoredist(ref_onsets, ref_pitches, est_onsets, est_pitches, pitch_mode='log'):
    cost_matrix = np.abs(est_onsets[:, np.newaxis] - ref_onsets[np.newaxis, :])
    est_idx, ref_idx = linear_sum_assignment(cost_matrix) # scipy

    # CALCULATE d_t (Temporal Distance)
    # Mean distance between aligned onsets
    matched_diffs_t = np.abs(est_onsets[est_idx] - ref_onsets[ref_idx])
    d_t = np.mean(matched_diffs_t)
    
    # CALCULATE d_p (Pitch Distance)
    # Match the pitches based on the onset alignment
    matched_ref_p = ref_pitches[ref_idx]
    matched_est_p = est_pitches[est_idx]
    
    if pitch_mode == 'log':
        # Use log base 2 to represent octaves/cents (human-centric)
        # Avoid log(0) with a small epsilon
        d_p = np.mean(np.abs(np.log2(matched_est_p + 1e-6) - np.log2(matched_ref_p + 1e-6)))
    else:
        # Raw Hz difference
        d_p = np.mean(np.abs(matched_est_p - matched_ref_p))
        
    return d_t, d_p, est_idx, ref_idx

# Extract onsets and melodies from an audio
#   audio: input audio samples, (samples,)
#   sr: sampling rate
#   hop_length: hop length
#   delta: onset threshold
#   min_melody_dur: minimum melody duration (sec)
#   n_max: maximum number of hamonic pitches
#   n_fft: fft size
# Return:
#   onsets_o: onsets in seconds, (onsets,)
#   f0_pitches_o: f0 pitches, (onsets,)
#   melodies_o: melodies of each onset, [[(hz, amp), .], ... ]
#   durations_o: duration (sec) of each melody, (onsets,)
#   audio_clips_o: samples of each melody, [(samples,), ... ]
def audiometa(audio, sr, 
              hop_length=512, 
              delta=0.01, 
              min_melody_dur=0.0, 
              dur_thres=0.0,
              n_max = 0, 
              n_fft=512):    
    onsets, envelope = audioonsets(audio, sr, hop_length=hop_length, delta=delta)
    audio_clips, onsets_d, clip_slices = audiosplitonset(audio, sr, onsets, min_melody_dur=min_melody_dur)  
    dur_t = audio.size/sr
    n_thres = int(dur_thres * sr)
    durations = np.concatenate((onsets_d, [dur_t - onsets[-1]]))
    melodies = melodyextractall(audio_clips, sr=sr, n_max=n_max, n_fft=n_fft)   

    onsets_o = []
    melodies_o = []
    durations_o = []
    audio_clips_o = []
    clip_slices_o = []
    for i, (onset_i, melody_i, dur_i, clip_i, slc_i) in enumerate(zip(onsets, melodies, durations, audio_clips, clip_slices)):
        if len(melody_i) > 0 and clip_i.size > n_thres:
            onsets_o.append(onset_i)
            melodies_o.append(melody_i)
            durations_o.append(dur_i)
            audio_clips_o.append(clip_i)
            clip_slices_o.append(slc_i)
    onsets_o = np.array(onsets_o)
    durations_o = np.array(durations_o)
    f0_pitches_o = np.array([p[0][0] for p in melodies_o])
    return onsets_o, f0_pitches_o, melodies_o, durations_o, audio_clips_o, clip_slices_o


# Onset detection
# hpss_i >=0 with ith harmonic and percussive component .
#       None, default, using the whole audio data.    
def audioonsets(audio, sr, hop_length=512, delta=0.01, hpss_i=None):
    audio_in = audio if hpss_i is None else librosa.effects.hpss(hpss)[hpss_i] # HPSS Separation
    onset_env = librosa.onset.onset_strength(y=audio_in, sr=sr, 
                                            hop_length=hop_length,
                                            aggregate=np.median)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, 
                                             sr=sr, 
                                             hop_length=hop_length,
                                             backtrack=True,
                                             delta=delta)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times, onset_env 


 

# Splitting an audio with onsets
# audio: input audio (samples,)
# sr: samling rate in hz
# onsets: onsets (seconds,)
# min_melody_dur: minimum melody duration in second, default 0.0.
# return: 
#     audio clips
#     onset differences
#     clip slices
def audiosplitonset(audio_t, sr, onsets, min_melody_dur=0.0, toend=True):
    onsets_d = np.diff(onsets)
    melodies_samples = (onsets_d*sr).astype(np.int32)
    min_melody_samples = int(min_melody_dur * sr) 
    k = int(onsets[0]*sr)  
    audio_clips = []
    clip_slices = []
    for i, kn in enumerate(melodies_samples):
        ke = k + max(kn, min_melody_samples)
        slc = slice(k, ke, None)
        audio_clip = audio_t[slc]
        #print('clip:', i, k, ke, audio_clip.size)
        audio_clips.append(audio_clip)
        clip_slices.append(slc)
        k += kn
    if toend and k < len(audio_t):
        slc = slice(k, None, None)
        audio_clips.append(audio_t[slc])
        clip_slices.append(slc)
    return audio_clips, onsets_d, clip_slices
    

def upboundi(T, t):
    n = len(T)
    v = bisect.bisect_left(T, t)
    v = np.clip(v, 0, n-1)
    return v  

# V2:Initializing parameters of audio elements with melody extraction
# Parameters:
#   onsets, melodies, durations, audio_clips: the meta data of target audio.
#   sr: sample rate.
#   timestamps_ele: the onset timestamps of element without the start point (time is 0.0).
#   clip_slices_ele: the slices of element 
#   gamma: the maxmimum stech rate.
#   strech_ratio: strench ratio.
#   max_strech_ratio: the maximum strench ratio.
#   dur_thres: the minimum duration of target clip.
# Return:
#   x = [tau, d, r, p, g, s]  
#   strech_ratio: strech ratio
#   sigma: x's sigma
#   dur_thres: duration thresolhd to eliminate a short melody
#   s: slice object of element clips.
def initxmelody(onsets, melodies, durations, audio_clips, sr, 
                timestamps_ele, clip_slices_ele,
                gamma=1.5,
                strech_ratio=1.0, max_strech_ratio=1.5,
                dur_thres = 0.0):   
    X = [] 
    m = len(timestamps_ele)
    #print('initxmelody::timestamps_ele', timestamps_ele)
    k = 0
    u = 0
    t_u = 0.0
    u_ = m - 1
    n_elements = 0
    for i, (tau_i, melodies_i, d_i, audio_clip) in enumerate( zip(onsets, melodies, durations, audio_clips) ):
        if d_i <= dur_thres:
            continue        
        te = d_i * gamma   
        ti = t_u + te
        if u < (u_):
            v = upboundi(timestamps_ele, ti)
        else:
            v = u
        t_v = timestamps_ele[v]
        ele_dur = t_v - t_u
        r_i = (ele_dur / d_i) * strech_ratio        
        p_i = melodies_i[:, 0]
        g_i = melodies_i[:, 1]

        x_i = [float(tau_i), float(d_i), float(r_i), p_i, g_i, float(t_u), float(t_v), int(u), int(v)]
        #print(f'Appending x[{k}]:')
        #print('ti', ti)            
        #print('u, v:', u, v)
        #print('ele_dur:', ele_dur)
        #print('t_u, t_v:', t_u, t_v)
        X.append(x_i) 
        if u == 0:
            n_elements += 1
        k += 1        
        t_u = timestamps_ele[u]
        u = v + 1
        if u >= m:
            u = 0
            t_u = 0.0

    return X, n_elements

# Splitting an audio 
# audio: input audio (samples,)
# sr: samling rate in hz
# dur_clip: duration of each clip
# return: 
#     a list audio clips
#     #samples of all clips, (clips,)
#     durations of all clips, (clips,)
def audiosplit(audio, sr, dur_clip):
    n_smp = audio.shape[0]
    dur = n_smp / sr    
    n_smp_clip = sr * dur_clip
    n_clips = int(np.ceil(n_smp / n_smp_clip))
    audio_clips = np.array_split(audio, n_clips)
    n_smp_clips = np.array([audio_clip.size for audio_clip in audio_clips])
    dur_clips = n_smp_clips / sr
    return audio_clips, n_smp_clips, dur_clips

# Synthesizing all melodies to an audio
# melodies:List of (freq, magnitude): [(f, m), ...]
# return: audio
def melody2audio(melodies, sr=22050, duration=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)
    for i, (freq, mag) in enumerate(melodies):
        envelope = np.sin(np.pi * t / duration) 
        audio_i = mag * np.sin(2 * np.pi * freq * t) * envelope
        audio += audio_i
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio
    
# Synthesizing all melodies to an audio
# melodies: a 2D list of melody sets: [[(f, m), ...], [(f, m), ...], ... ]
# return: audio
def melodies2audio(melodies, sr, durations):
    audio_out = []     
    for i, (melody, dur) in enumerate(zip(melodies, durations)): 
        audio_i = melody2audio(melody, sr, dur)
        audio_out.append(audio_i)
    audio_out = np.concatenate(audio_out)    
    return audio_out

# Extracing all melodies from an audio clip
# return: 
#     list of melodies:[(freq, mag),...], 
def melodyextract(audio, sr, 
                n_max=0,
                bins_per_octave=24,
                n_bins=24*7,
                peak_height=0.05,
                peak_distance=12,
                n_fft=512,
                fmin=librosa.note_to_hz('C1'),
                harmonic=True, harmonic_tol=0.03):    
    if audio.size == 0:
        return []
    #print('melodyextract::cqt')
    cqt = np.abs(librosa.cqt(audio, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins))
    mean_cqt = np.mean(cqt, axis=1)
    #print('melodyextract::find_peaks')
    peaks, properties = scipy.signal.find_peaks(mean_cqt, height=peak_height, distance=peak_distance)
    peak_heights = properties['peak_heights']
    #print('melodyextract::cqt_frequencies')
    freqs = librosa.cqt_frequencies(n_bins=len(mean_cqt), fmin=fmin, bins_per_octave=bins_per_octave)
    peak_freqs = freqs[peaks] 
    sorted_idx = np.argsort(peak_heights)[::-1]
    melodies = list(zip(peak_freqs[sorted_idx], peak_heights[sorted_idx]))
    if harmonic:
        melodies_flt = []
        for freq, mag in melodies:
            is_harmonic = False
            for existing_f, _ in melodies_flt:
                ratio = freq / existing_f
                if abs(ratio - round(ratio)) < harmonic_tol: 
                    is_harmonic = True
                    break            
            if not is_harmonic:
                melodies_flt.append((freq, mag))      
        melodies = melodies_flt
    n_max = None if (n_max is not None) and (n_max<=0) else n_max
    melodies = np.array(melodies[:n_max])
    #print('melodyextract::OK')
    return melodies

# Extracing all melodies from an audio
# return: 
#     list of [(f, m),...], melody set of each audio clip : 
#              [[(f, m), ...], [(f, m), ...], ... ]
def melodyextractall(audio_clips, sr, 
                n_max=0,
                bins_per_octave=24,
                n_bins=24*7,
                peak_height=0.05,
                peak_distance=12,
                n_fft=512,
                fmin=librosa.note_to_hz('C1'),
                harmonic=True, harmonic_tol=0.03): 
    melodies = []
    for i, audio_clip in enumerate(audio_clips):
        melodies_i = melodyextract(audio_clip, sr, 
                                   n_max=n_max,
                                   bins_per_octave=bins_per_octave,
                                   n_bins=n_bins,
                                   peak_height=peak_height,
                                   peak_distance=peak_distance,
                                   n_fft=n_fft,
                                   fmin=fmin,
                                   harmonic=harmonic,
                                   harmonic_tol=harmonic_tol)
        melodies.append(melodies_i)    
    return melodies
# melodyextractall

# Dividing an audio by beat. 
# Note that the final clip is the interval the last beat and the last samples.
#   audio: input audio
#   sr: sampling rate
#   hop_length: frame size
# Return: 
#   tempo (float)
#   audio_clips: (list), a list of audio clips
#   bbs: (list), the bounding boxe of each clip.
#   dur: (ndarray), the duration of each clip 
#   beat_times: (ndarray), the start time of each clip 
def beatclip(audio, sr, hop_length=512):
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    if len(beat_frames) == 0:
        tempo = 1
        beat_samples = np.array([0])
    beat_samples = librosa.frames_to_samples(beat_frames, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, hop_length=hop_length, sr=sr)
    n_beats = len(beat_frames)
    audio_clips = []
    bbs = []
    n_samples = []
    n_audio_smp = len(audio)
    beat_samples = np.concatenate([beat_samples, [n_audio_smp]])                             
    i = beat_samples[0]
    for j in beat_samples[1:]:
        audio_clips.append(audio[i:j])
        bbs.append(slice(i, j, None))
        n_samples.append(j - i)
        i = j     
    n_samples = np.array(n_samples)
    durations = n_samples / sr
    return tempo, audio_clips, bbs, durations, beat_times


# audio_a -> audio_b
def majorpitch(audio_a, audio_b, sr, pmin='C2', pmax='C7'):
    f0_a, voiced_flag_a, voiced_probs_a = librosa.pyin(audio_a, sr=sr, fmin=librosa.note_to_hz(pmin), fmax=librosa.note_to_hz(pmax))
    f0_b, voiced_flag_b, voiced_probs_b = librosa.pyin(audio_b, sr=sr, fmin=librosa.note_to_hz(pmin), fmax=librosa.note_to_hz(pmax))
    voiced_f0_a = f0_a[voiced_flag_a]
    voiced_f0_b = f0_b[voiced_flag_b]
    major_pitch_hz_a = None
    major_pitch_hz_b = None
    n_steps = 0
    if len(voiced_f0_a) > 0:
        major_pitch_hz_a = np.median(voiced_f0_a)
    if len(voiced_f0_b) > 0:
        major_pitch_hz_b = np.median(voiced_f0_b)
    if (major_pitch_hz_a is not None) and (major_pitch_hz_b is not None):
        n_steps = 12 * np.log2(major_pitch_hz_b / major_pitch_hz_a)
    return n_steps, major_pitch_hz_a, major_pitch_hz_b

# Get the intensity ranges of audio clips.
#  audio_clips: audio clips
def intrange(audio_clips):    
    intrange = np.array([ [audio_i.min(),  audio_i.max()] for audio_i in audio_clips])
    return intrange
         

# stretching an audio 
#  rate: stretch rate
def audiostretch(audio, rate, n_fft=2048):
    audio_scaled = librosa.effects.time_stretch(audio, rate=rate, n_fft=n_fft)
    return audio_scaled

# resaampling
def audioscale(audio, sr, scale):
    sr_scale = int(sr * scale)
    audio_tr = librosa.resample(y=audio, orig_sr=sr, target_sr=sr_scale)
    return audio_tr
 
# Creating Gaussian weight
#  _audio: audio data
#  sigma: sigma
def creategw(audio, sigma):
    n = len(audio)
    x = np.linspace(-1, 1, n)
    y = -((x / sigma) ** 2.0) / 2
    g = np.exp(y)
    return g

 

# Mixing 
#   audio_base += audio or 
#   t: start time (s)
#   sr: sample rate (Hz)
def mix(audio_base, audio, t, sr, unmix=False):
    i = int(t * sr)
    j = np.clip(i+len(audio), 0, len(audio_base))
    n = j - i    
    bb_b = slice(i, j, None)
    bb_a = slice(0, n, None)
    if unmix:
        audio_base[bb_b] = audio_base[bb_b] - audio[bb_a] 
    else:
        audio_base[bb_b] = audio_base[bb_b] + audio[bb_a] 
    return audio_base 

# Mixing a seires of audio elements
#   audio_out: Output audio, ndarray
#   audio = original audio element, ndarray
#   q: rendering parameters
#   sr: sample rate (Hz)
def mixq(audio_out, audio, q, sr, unmix = False, n_fft=2048,
         fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'), 
         sw=[True, True, True, True, True], dim=5):
    audio_q = audio
    d, r, p, sigma, g, tau = q
    if sw[0] and dim>=0:
        audio_q = audiostretch(audio_q, r, n_fft=n_fft)

    if sw[1] and dim>=1:
        audio_q = psola.vocode(audio_q, sample_rate=sr, 
                             target_pitch=[p,], 
                             fmin=fmin, fmax=fmax)

    if sw[2] and dim>=2:
        gw = creategw(audio_q, sigma)
        audio_q = audio_q * gw

    if sw[3] and dim>=3:
        audio_q = audio_q * g

    if sw[4] and dim>=4:
        audio_out = mix(audio_out, audio_q, tau, sr, unmix=unmix)

    return audio_out, audio_q

#  x = [tau, d, r, p, g, t_u, t_v, u, v]  
#  q = [d, r, p, sigma, gt, tau] 
def mixx(audio_out, audio, x, sr, sigma=1.0, unmix = False, n_fft=2048,
         fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'), 
         sw=[True, True, True, True, True], dim=5):
    tau_i, d_i, r_i, pitches_i, gains_i, t_u, t_v, u, v = x
    slc_i = slice(int(t_u*sr), int(t_v*sr), None)
    audio_ele_i = audio[slc_i]
    for p_i, g_i in zip(pitches_i, gains_i):        
        q_i = np.array([d_i, r_i, p_i, sigma, g_i, tau_i])
        audio_out, _ = mixq(audio_out, audio_ele_i, q_i, sr, 
                       unmix=unmix, n_fft=n_fft, fmin=fmin, fmax=fmax, sw=sw, dim=dim)
    return audio_out

# ================================================
# Audio element synthesizer
class Synthesizer:
    def __init__(self, target_path, element_path, sr=22050, max_duration=None, 
                    pitch_mode='log', # for scoredist()
                ):
        self.target_path = target_path
        self.element_path = element_path
        self.sr = sr
        n_samples = None if max_duration is None else int(sr * max_duration)
        self.audio_t_org = librosa.load(target_path, sr=sr)[0][:n_samples]
        self.audio_t = self.audio_t_org
        self.audio_ele = librosa.load(element_path, sr=sr)[0]

        self.pitch_mode = pitch_mode 
 
        self.n_samples_out = self.audio_t.size
        self.n_samples_ele = self.audio_ele.size
        self.out_shape = (self.n_samples_out, )
        self.dur_out = self.n_samples_out / self.sr
        self.dur_ele = self.n_samples_ele / self.sr 
    # Synthesizer::__init__
 

    def melodydist(self, _x): # _x: (samples, )
        (onsets_x, 
         f0_pitches_x, 
         melodies_x, 
         durations_x, 
         audio_clips_x, 
         clip_slices_x) = audiometa(audio=_x, 
                                     sr=self.sr, 
                                     hop_length=self.hop_length, 
                                     delta=self.delta, 
                                     min_melody_dur=0.0, 
                                     n_max=self.n_max_clip, 
                                     n_fft=self.n_fft)      
   
        d_t, d_p, _, _ = scoredist(ref_onsets=self.onsets_t, 
                                   ref_pitches=self.f0_pitches_t, 
                                   est_onsets=onsets_x, 
                                   est_pitches=f0_pitches_x, 
                                   pitch_mode=self.pitch_mode)
        return d_t, d_p   
 
     
    # -------------------------------
    def synthesize(self, X=None, g_sigma=None,  
                    strech_ratio=0.6, sigma = 0.4, gamma=0.3,
                    hop_length = 512, delta=0.05, n_fft=512, 
                    min_melody_dur=0.5, dur_thres=0.2,
                    n_max_clip=0, 
                    delta_ele=0.02,
                    sw=[True, True, True, True, True],                    
                    verbose=1, n_log=1): 
        if verbose & 2:
            # Parameter listing
            args_info = inspect.getargvalues(inspect.currentframe())
            for name in args_info.args[1:]: # self not included
                print('synthesize::%s='%name, args_info.locals[name], sep='')

        self.hop_length = hop_length
        self.delta=delta
        self.delta_ele=delta_ele
        self.n_max_clip=n_max_clip
        self.n_fft=n_fft
        self.min_melody_dur=min_melody_dur
        self.sigma=sigma

 
        # Gaussian blurring the target
        self.audio_t = self.audio_t_org if g_sigma is None else gaussian_filter1d(self.audio_t_org, sigma=g_sigma)

        (self.onsets_t, 
        self.f0_pitches_t, 
        self.melodies_t, 
        self.durations_t, 
        self.audio_clips_t, 
        self.clip_slices_t) = audiometa(audio=self.audio_t, 
                                     sr=self.sr, 
                                     hop_length=hop_length, 
                                     delta=delta, 
                                     min_melody_dur=min_melody_dur, 
                                     dur_thres=dur_thres,
                                     n_max=n_max_clip, 
                                     n_fft=n_fft)  

        # v2: for multi-onset element
        (self.onsets_ele, 
         self.f0_pitches_ele, 
         self.melodies_ele, 
         self.durations_ele, 
         self.audio_clips_ele, 
         self.clip_slices_ele) = audiometa(audio=self.audio_ele, 
                                     sr=self.sr, 
                                     hop_length=hop_length, 
                                     delta=self.delta_ele, 
                                     min_melody_dur=0.0, 
                                     dur_thres=0.0,
                                     n_max=0, 
                                     n_fft=n_fft)

        # v2: creating concatnated slices and durations for the element audio                        
        n_onsets_ele = self.onsets_ele.shape[0]
        self.timestamps_ele = np.cumsum(self.durations_ele)
        self.timestamps_ele[-1] = self.dur_ele
        
        if X is None:
            X, self.n_elements = initxmelody(onsets=self.onsets_t, 
                                        melodies=self.melodies_t, 
                                        durations=self.durations_t, 
                                        audio_clips=self.audio_clips_t, 
                                        sr=self.sr, 
                                        timestamps_ele=self.timestamps_ele, 
                                        clip_slices_ele=self.clip_slices_ele,
                                        gamma=gamma,
                                        strech_ratio=strech_ratio, 
                                        dur_thres=dur_thres) 

        if verbose & 1:
            print('synthesize::|X|', len(X))
            print('synthesize::n_elements', self.n_elements)
 
        n_x = len(X)
        audio_out = np.zeros(self.out_shape)

        for i, x_i in enumerate(X):
            audio_out = mixx(audio_out, self.audio_ele, x_i, self.sr, sigma=sigma, sw=sw)
 
        return audio_out, X
    # Synthesizer::synthesize
    


# ====================================
def exit_program():    
    sys.exit(0)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    app_description = "SAMS v2.0"
    def_arg = {
        "t":"",                    # target audio filepath
        "s":"",                    # speech audio filepath        
        "o":"",                    # output audio filepath
        "gamma":0.3,               # gamma, the maximum allowable speedup rate        
        "r":0.6,                   # strech ratio
        "sr":22050,                # sample rate
        "d":None,                  # Maximum duration
    }    

    parser = ArgumentParser(description=app_description)
    parser.add_argument('t', type=str, default=def_arg["t"], help='target audio filepath')
    parser.add_argument('s', type=str, default=def_arg["s"], help='speech audio filepath')    
    parser.add_argument('o', type=str, default=def_arg["t"], help='output audio filepath')
    parser.add_argument('--gamma', type=float, default=def_arg["gamma"], help='gamma')    
    parser.add_argument('--r', type=float, default=def_arg["r"], help='strech ratio')
    parser.add_argument('--sr', type=int, default=def_arg["sr"], help='sample rate')
    parser.add_argument('--d', type=int, default=def_arg["d"], help='Maximum duration')
     
    args = parser.parse_args()
    

 
    if not os.path.isfile(args.s):
        print('Speech audio file not exist:', args.s)
        exit_program()
    if not os.path.isfile(args.t):
        print('Target audio file not exist:', args.t)
        exit_program()
 
    output_dir, output_filename, output_extname =  parsePath(args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tm_s = time()
    syn = Synthesizer(args.t, args.s, sr=args.sr, max_duration=args.d)        
    audio_out, X = syn.synthesize(gamma=args.gamma, strech_ratio=args.r)  
    sf.write(args.o, audio_out, args.sr)
    tm = time() - tm_s
    print(f'Synthesis time: {tm:0.2f}')
    exit_program()  

if __name__ == "__main__":
    main()           

# python sams.py target/Beatles_LetItBe.mp3 speech/hello.wav output/test.wav     
