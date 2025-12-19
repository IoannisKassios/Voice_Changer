import numpy as np
import sounddevice as sd
import time
from scipy import signal

class VoiceDistorter:
  def __init__(self, chunk_size=512, sample_rate=44100):
    self.pitch_shift = -0.08
    self.bit_crush_bits = 14
    self.ring_mod_freq = 12
    self.distortion_gain = 1.1
    self.noise_threshold = 0.008
    self.gate_release_time = 0.5
    sd.default.samplerate = self.sample_rate
    sd.default.channels = 1
    sd.default.blocksize = self.chunk_size

  def noise_gate(self, audio, threshold):
    rms = np.sqrt(np.mean(audio**2))
    if rms > threshold:
      return audio
    else:
      return audio * 0.02
    
  def light_but_crush(self, audio, bits):
    max_val = 2**(bits-1)
    crushed = np.round(audio * max_val) / max_val
    return 0.3 * crushed + 0.7 * audio
  
  def suble_ring_modulation(self, audio, freq):
    t = np.linspace(0, len(audio)/self.sample_rate, len(audio), endpoint = False)
    modulator = 0.8 + 0.2 * np.sin(2 * np.pi * freq * t)
    return audio * modulator
  
  def pitch_shift_simple(self, audio, shift_factor):
    indices = np.linspace(0, len(audio) - 1, int(len(audio)/(1+shift_factor)))
    return np.interp(indices, np.arange(len(audio)), audio)[:len(audio)]
  
  def apply_distortion(self, audio):
    audio = self.noise_gate(audio, self.noise_threshold)
    audio = self.pitch_shift_simple(audio, self.pitch_shift)
    audio = np.tanh(audio * self.distortion_gain) * 0.95
    audio = self.light_but_crush(audio, self.bit_crush_bits)
    audio = self.suble_ring_modulation(audio, self.ring_mod_freq)
    sos_high = signal.butter(1, 4000, 'high', fs=self.sample_rate, output = 'sos')
    sos_low = signal.butter(1, 4000, 'low', fs=self.sample_rate, output = 'sos')
    return np.clip(audio, -1.0, 1.0)
  
  def start(self):
    with sd.Stream(callback=self.audio_callback):
      while True:
        time.sleep(0.1)
  
  if __name__ == "__main__":
    VoiceDistorter().start()