import torch
import torch.nn as nn
import numpy as np
import librosa
import cv2
import os
import pickle
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """多模态特征提取器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._init_text_extractor()
        self._init_audio_extractor()
        self._init_video_extractor()

    def _init_text_extractor(self):
        """初始化文本特征提取器"""
        print("初始化BERT文本特征提取器...")
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.text_model = BertModel.from_pretrained('bert-base-chinese')
        self.text_model.to(self.device)
        self.text_model.eval()

    def _init_audio_extractor(self):
        """初始化音频特征提取器"""
        print("初始化音频特征提取器...")
        # 使用Wav2Vec2进行音频特征提取
        try:
            self.audio_processor = Wav2Vec2Processor.from_pretrained(
                "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
            self.audio_model = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
            self.audio_model.to(self.device)
            self.audio_model.eval()
            self.use_wav2vec2 = True
        except:
            print("Wav2Vec2加载失败，使用传统音频特征提取")
            self.use_wav2vec2 = False

    def _init_video_extractor(self):
        """初始化视频特征提取器"""
        print("初始化视频特征提取器...")
        # 使用ResNet进行视觉特征提取
        self.video_model = models.resnet50(pretrained=True)
        self.video_model = nn.Sequential(*list(self.video_model.children())[:-1])  # 移除最后的全连接层
        self.video_model.to(self.device)
        self.video_model.eval()

        # 图像预处理
        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extract_text_features(self, text):
        """
        提取文本特征

        参数:
            text: 输入文本字符串

        返回:
            features: 文本特征向量
        """
        try:
            with torch.no_grad():
                # BERT特征提取
                inputs = self.text_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.text_model(**inputs)
                # 使用[CLS] token的表示作为文本特征
                text_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                return text_features[0]  # 返回768维特征

        except Exception as e:
            print(f"文本特征提取错误: {e}")
            return np.zeros(768)  # 返回零向量作为备用

    def extract_audio_features_wav2vec2(self, audio_path):
        """
        使用Wav2Vec2提取音频特征

        参数:
            audio_path: 音频文件路径

        返回:
            features: 音频特征向量
        """
        try:
            if not os.path.exists(audio_path):
                return np.zeros(1024)

            # 加载音频
            speech, sr = librosa.load(audio_path, sr=16000)

            # 处理音频
            inputs = self.audio_processor(
                speech,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                # 取最后一层隐藏状态的均值作为特征
                audio_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                return audio_features[0]  # 返回1024维特征

        except Exception as e:
            print(f"Wav2Vec2音频特征提取错误: {e}")
            return self.extract_audio_features_traditional(audio_path)

    def extract_audio_features_traditional(self, audio_path):
        """
        使用传统方法提取音频特征

        参数:
            audio_path: 音频文件路径

        返回:
            features: 音频特征向量 (33维)
        """
        try:
            if not os.path.exists(audio_path):
                return np.zeros(33)

            # 加载音频
            y, sr = librosa.load(audio_path, sr=22050)

            features = []

            # 1. MFCC特征 (13维)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            features.extend(mfcc_mean)

            # 2. 基频特征
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
            f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
            features.append(f0_mean)

            # 3. 频谱特征
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroid)
            features.append(spectral_centroid_mean)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            features.append(spectral_rolloff_mean)

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            features.append(spectral_bandwidth_mean)

            # 4. 过零率
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            features.append(zcr_mean)

            # 5. 能量特征
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)
            features.append(rms_mean)

            # 6. 色度特征 (12维)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)

            # 确保特征维度为33
            features = features[:33]
            if len(features) < 33:
                features.extend([0] * (33 - len(features)))

            return np.array(features)

        except Exception as e:
            print(f"传统音频特征提取错误: {e}")
            return np.zeros(33)

    def extract_audio_features(self, audio_path):
        """提取音频特征（自动选择方法）"""
        if self.use_wav2vec2:
            return self.extract_audio_features_wav2vec2(audio_path)
        else:
            return self.extract_audio_features_traditional(audio_path)

    def extract_video_features(self, video_path, num_frames=10):
        """
        提取视频特征

        参数:
            video_path: 视频文件路径
            num_frames: 采样的帧数

        返回:
            features: 视频特征向量
        """
        try:
            if not os.path.exists(video_path):
                return np.zeros(2048)  # ResNet50特征维度

            # 打开视频
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return np.zeros(2048)

            # 计算采样间隔
            interval = max(1, total_frames // num_frames)

            frame_features = []
            frames_processed = 0

            for i in range(0, total_frames, interval):
                if frames_processed >= num_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    continue

                # 提取帧特征
                frame_feature = self._extract_frame_features(frame)
                if frame_feature is not None:
                    frame_features.append(frame_feature)
                    frames_processed += 1

            cap.release()

            if len(frame_features) > 0:
                # 对所有帧特征取平均
                video_features = np.mean(frame_features, axis=0)
                return video_features
            else:
                return np.zeros(2048)

        except Exception as e:
            print(f"视频特征提取错误: {e}")
            return np.zeros(2048)

    def _extract_frame_features(self, frame):
        """提取单帧图像特征"""
        try:
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 人脸检测
            faces = self.face_cascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                # 使用最大的人脸区域
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face_roi = frame_rgb[y:y + h, x:x + w]
            else:
                # 如果没有检测到人脸，使用整个帧
                face_roi = frame_rgb

            # 调整大小并预处理
            pil_image = Image.fromarray(face_roi)
            input_tensor = self.video_transform(pil_image).unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                features = self.video_model(input_tensor)
                features = features.squeeze().cpu().numpy()

            return features

        except Exception as e:
            print(f"帧特征提取错误: {e}")
            return None

    def extract_all_features(self, text, audio_path, video_path):
        """
        提取所有模态的特征

        参数:
            text: 文本
            audio_path: 音频路径
            video_path: 视频路径

        返回:
            features_dict: 包含所有特征的字典
        """
        features_dict = {}

        # 提取文本特征
        features_dict['text'] = self.extract_text_features(text)

        # 提取音频特征
        features_dict['audio'] = self.extract_audio_features(audio_path)

        # 提取视频特征
        features_dict['video'] = self.extract_video_features(video_path)

        return features_dict


class CHSIMSFeaturePreprocessor:
    """CH-SIMS数据集特征预处理器"""

    def __init__(self, data_dir, output_dir, device='auto'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化特征提取器
        self.extractor = FeatureExtractor(self.device)

        # 数据路径
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.metadata_path = os.path.join(data_dir, 'meta.csv')

    def load_metadata(self):
        """加载元数据"""
        self.metadata = pd.read_csv(self.metadata_path)
        return self.metadata

    def preprocess_all_features(self):
        """预处理所有特征"""
        print("开始预处理CH-SIMS数据集特征...")

        # 加载元数据
        metadata = self.load_metadata()

        features_data = {}

        for idx, row in metadata.iterrows():
            video_id = row['video_id']
            text = row['text']

            print(f"处理样本 {idx + 1}/{len(metadata)}: {video_id}")

            try:
                # 构建文件路径
                audio_path = os.path.join(self.raw_dir, 'audio', f"{video_id}.wav")
                video_path = os.path.join(self.raw_dir, 'video', f"{video_id}.mp4")

                # 提取特征
                features = self.extractor.extract_all_features(text, audio_path, video_path)
                features_data[video_id] = features

            except Exception as e:
                print(f"处理样本 {video_id} 时出错: {e}")
                # 创建空特征作为备用
                features_data[video_id] = {
                    'text': np.zeros(768),
                    'audio': np.zeros(1024 if self.extractor.use_wav2vec2 else 33),
                    'video': np.zeros(2048)
                }

        # 保存特征数据
        output_path = os.path.join(self.output_dir, 'processed_features.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(features_data, f)

        print(f"特征预处理完成！保存至: {output_path}")

        return features_data

    def create_aligned_dataset(self):
        """创建对齐的数据集"""
        # 加载特征
        features_path = os.path.join(self.output_dir, 'processed_features.pkl')
        if not os.path.exists(features_path):
            print("未找到预处理的特征文件，请先运行 preprocess_all_features()")
            return None

        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)

        # 加载元数据
        metadata = self.load_metadata()

        aligned_data = {}

        for _, row in metadata.iterrows():
            video_id = row['video_id']

            if video_id in features_data:
                aligned_data[video_id] = {
                    'text': features_data[video_id]['text'],
                    'audio': features_data[video_id]['audio'],
                    'video': features_data[video_id]['video'],
                    'label': row['label'],
                    'label_T': row.get('label_T', row['label']),
                    'label_A': row.get('label_A', row['label']),
                    'label_V': row.get('label_V', row['label']),
                    'annotation': row['annotation']  # 修复：使用正确的列名
                }

        # 保存对齐数据
        aligned_path = os.path.join(self.output_dir, 'aligned_features.pkl')
        with open(aligned_path, 'wb') as f:
            pickle.dump(aligned_data, f)

        print(f"对齐数据集创建完成！保存至: {aligned_path}")

        return aligned_data


# 使用示例
if __name__ == '__main__':
    # 初始化预处理器
    preprocessor = CHSIMSFeaturePreprocessor(
        data_dir='data',
        output_dir='data/processed'
    )

    # 预处理所有特征
    features_data = preprocessor.preprocess_all_features()

    # 创建对齐数据集
    aligned_data = preprocessor.create_aligned_dataset()

    print("特征提取完成！")