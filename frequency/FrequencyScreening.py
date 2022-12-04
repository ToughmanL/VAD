import librosa
import numpy as np
import random

class WavFreqScreen:
  def __init__(self):
    self.wlen = 256
    self.inc = 128

  # 计算每帧对应的时间
  def _FrameTimeC(self, frameNum, frameLen, inc, fs):
      ll = np.array([i for i in range(frameNum)])
      return ((ll - 1) * inc + frameLen / 2) / fs

  # 分帧函数
  def _Enframe(self, x, win, inc=None):
      nx = len(x)
      if isinstance(win, list) or isinstance(win, np.ndarray):
          nwin = len(win)
          nlen = nwin  # 帧长=窗长
      elif isinstance(win, int):
          nwin = 1
          nlen = win  # 设置为帧长
      if inc is None:
          inc = nlen
      nf = (nx - nlen + inc) // inc
      frameout = np.zeros((nf, nlen))
      indf = np.multiply(inc, np.array([i for i in range(nf)]))
      for i in range(nf):
          frameout[i, :] = x[indf[i]:indf[i] + nlen]
      if isinstance(win, list) or isinstance(win, np.ndarray):
          frameout = np.multiply(frameout, np.array(win))
      return frameout

  #加窗
  def _HanningWindow(self, N):
      nn = [i for i in range(N)]
      return 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))

  # 短时傅里叶变换
  def _STFFT(self, x, win, nfft, inc):
      xn = self._Enframe(x, win, inc)
      xn = xn.T
      y = np.fft.fft(xn, nfft, axis=0)
      return y[:nfft // 2, :]

  # 筛选代码
  def Screen(self, path):
    true_freq = 16
    wlen = self.wlen
    nfft = wlen
    win = self._HanningWindow(wlen)
    inc = self.inc
    data_all, fs = librosa.load(path, sr=None, mono=False)#sr=None声音保持原采样频率， mono=False声音保持原通道数
    if data_all.shape[0] > 16000:
      temp = random.randint(0, data_all.shape[0]-16000)
      data = data_all[:16000]
    else:
      data = data_all
    y = self._STFFT(data, win, nfft, inc)
    # FrequencyScale = [i * fs / wlen for i in range(wlen // 2)] #频率刻度
    # frameTime = self._FrameTimeC(y.shape[1], wlen, inc, fs) #每帧对应的时间
    LogarithmicSpectrogramData=10*np.log10((np.abs(y)*np.abs(y))) #取对数后的数据

    sum = float('-inf')
    last_sum = float('-inf')
    for i in range(LogarithmicSpectrogramData.shape[1]):
      if i < 70 or i > 125:
        continue
      temp_spec = np.abs(LogarithmicSpectrogramData[i])
      sum = np.sum(temp_spec)
      if last_sum == float('-inf'):
        last_sum = sum
        continue
      if abs(sum-last_sum) > 1000 or abs(sum-last_sum) > sum*0.16:
        true_freq = i/8
      last_sum = sum
    return true_freq

path=r"/home/toughman/WorkSpace/VAD/frequency/4_01.wav"#audio002.wav
test_screen = WavFreqScreen()
true_freq = test_screen.Screen(path)
print(true_freq)