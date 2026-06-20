import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

# ==========================================
# 1. 加载 3 个麦克风的音频数据
# ==========================================
# 请确保这三个文件名与你电脑上的实际文件名完全一致
audio_path1 = r'D:\LSL\神经网络课程\附件\附件\BSS\110000001mix1.wav'
audio_path2 = r'D:\LSL\神经网络课程\附件\附件\BSS\110000001mix2.wav'
audio_path3 = r'D:\LSL\神经网络课程\附件\附件\BSS\110000001mix3.wav'  

print("正在加载 3 个通道的音频...")
try:
    audio1, sample_rate = sf.read(audio_path1)
    audio2, _ = sf.read(audio_path2)
    audio3, _ = sf.read(audio_path3)
    
    # 确保三个音频长度完全一致，截断到最短的那个
    min_len = min(len(audio1), len(audio2), len(audio3))
    
    # 将 3 个单声道文件堆叠成一个 (样本数, 3) 的矩阵
    mixed_audio = np.vstack((audio1[:min_len], audio2[:min_len], audio3[:min_len])).T
    print(f"音频合并成功！采样率: {sample_rate}Hz, 矩阵形状: {mixed_audio.shape}")
except Exception as e:
    print(f"加载音频失败，请检查文件名是否写对。报错详情: {e}")
    exit()

n_samples, n_channels = mixed_audio.shape

# ==========================================
# 2. 算法一：FastICA 分离 (提取 3 个独立声源)
# ==========================================
print("正在使用 FastICA 进行源分离...")
ica = FastICA(n_components=n_channels, random_state=42)
ica_signals = ica.fit_transform(mixed_audio)

# 归一化防止爆音
ica_signals = ica_signals / np.max(np.abs(ica_signals)) * 0.9

for i in range(n_channels):
    sf.write(f'ica_separated_source_{i+1}.wav', ica_signals[:, i], sample_rate)

# ==========================================
# 3. 算法二：PCA 分离 (作为对比)
# ==========================================
print("正在使用 PCA 进行源分离...")
pca = PCA(n_components=n_channels, random_state=42)
pca_signals = pca.fit_transform(mixed_audio)

pca_signals = pca_signals / np.max(np.abs(pca_signals)) * 0.9

for i in range(n_channels):
    sf.write(f'pca_separated_source_{i+1}.wav', pca_signals[:, i], sample_rate)

# ==========================================
# 4. 可视化评估 (展示 FastICA 成功分离的 3 个声音)
# ==========================================
print("正在生成波形图与语谱图对比...")
# 创建 3行2列 的图表，分别展示 3 个声源的波形和语谱图
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

titles = ['Source 1', 'Source 2', 'Source 3']
colors = ['blue', 'green', 'purple']

for i in range(3):
    # 绘制波形图
    axes[i, 0].plot(ica_signals[:, i], color=colors[i])
    axes[i, 0].set_title(f'FastICA Separated {titles[i]} (Waveform)')
    axes[i, 0].set_xlabel('Samples')
    axes[i, 0].set_ylabel('Amplitude')
    
    # 绘制语谱图
    axes[i, 1].specgram(ica_signals[:, i], Fs=sample_rate, cmap='viridis')
    axes[i, 1].set_title(f'FastICA Separated {titles[i]} (Spectrogram)')
    axes[i, 1].set_xlabel('Time (s)')
    axes[i, 1].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

print("处理完成！请去文件夹里听一下 ica_separated 和 pca_separated 的 wav 文件进行对比。")
