# SAMS: A Low-Complexity Speech-to-Music Synthesis Framework via Syllable-Level Alignment
SAMS is a lightweight framework for speech-to-music alignment. By mapping speech syllables to a track's pitch and rhythm using deterministic transformations, it ensures intelligibility, low latency, and low computational overhead.

## Usage
```python
python sams t s o [--gamma 0.3]
```
- t: target audio filepath
- s: speech audio filepath        
- o: output audio filepath
gamma:  $\gamma$, the maximum allowable speedup rate

You can download the test audio files from the following links:
1. [Let It Be - The Beatles](https://www.audiolabs-erlangen.de/resources/MIR/2015-ISMIR-LetItBee/)
2. [ADC2004](http://labrosa.ee.columbia.edu/projects/melody/)
3. [FMA](https://github.com/mdeff/fma)
