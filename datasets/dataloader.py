import os
import torch
import csv
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

import torchaudio
import torchaudio.transforms as audio_T
from PIL import Image, ImageFilter
from transformers import AutoFeatureExtractor
import sys
import warnings
warnings.filterwarnings("ignore")
import random
# torchaudio.set_audio_backend("sox_io")
sys.path.append('./datasets/')


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
 
        data = []
        self.args = args
        self.has_annotations = False
        self.annotation_type = None
        self.sample_layout = None

        if args.dataset_mode != 'VGGSound':
            raise NotImplementedError(
                f'Dataset mode "{args.dataset_mode}" not implemented')

        # Debug with a small dataset
        if args.debug:
            
            if mode=='train':
                with open('metadata/debug_data/train_vggs_debug_100.txt','r') as f:
                    txt_reader = f.readlines()
                    for item in txt_reader:
                        data.append(item.rstrip('\n'))
                    self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                    self.video_path = args.trainset_path + '/total_video_frames/'
                    self.sample_layout = 'vggs_train'
            
            elif mode=='test':
                with open('metadata/debug_data/test_vggss_debug_50.txt','r') as f:
                    txt_reader = f.readlines()
                    for item in txt_reader:
                        data.append(item.split('.')[0])
                    self.audio_path = args.vggss_test_path + '/audio/'
                    self.video_path = args.vggss_test_path + '/frame/'
                    self.has_annotations = True
                    self.annotation_type = 'vggss'
                    self.sample_layout = 'flat_image'

            elif mode=='val':
                with open('metadata/test_flick.csv') as f:
                    csv_reader = csv.reader(f)
                    for item in csv_reader:
                        data.append(item[0])
                    
                    self.audio_path = args.soundnet_test_path + '/audio/'
                    self.video_path = args.soundnet_test_path + '/frames/'
                    self.has_annotations = True
                    self.annotation_type = 'soundnet'
                    self.sample_layout = 'flat_image'

        else:
            if args.dataset_mode == 'VGGSound':
                if mode=='train':
                    if self.args.training_set_scale == 'subset_144k':
                        if self.args.ret_seen_144k:
                            train_list_file = 'train_seen_144k_list.txt'
                        else:
                            train_list_file = 'train_vggs_144k.txt'
                    elif self.args.training_set_scale == 'subset_143k':
                        train_list_file = 'train_vggs_143k.txt'
                    elif self.args.training_set_scale == 'subset_10k':
                        train_list_file = 'train_vggs_10k.txt'
                    elif self.args.training_set_scale == 'subset_1k':
                        train_list_file = 'train_vggs_1k.txt'
                    else:
                        train_list_file = 'train_vggs_190228.txt' 

                    with open('metadata/' + train_list_file,'r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.rstrip('\n'))
                        self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                        self.video_path = args.trainset_path + '/total_video_frames/'
                        self.sample_layout = 'vggs_train'

                elif mode=='test':
                    if self.args.testing_set_scale == 'subset_250':
                        test_list_file = 'test_vggss_250.txt'
                    else:
                        test_list_file = 'test_vggss_4911.txt'
                    
                    with open('metadata/' + test_list_file, 'r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader[:]:
                            data.append(item.split('.')[0])
                        self.audio_path = args.vggss_test_path + '/audio/'
                        self.video_path = args.vggss_test_path + '/frame/'
                        self.has_annotations = True
                        self.annotation_type = 'vggss'
                        self.sample_layout = 'flat_image'
                
                elif mode=='val':
                    if self.args.val_set == 'VGGS':
                        if self.args.val_set_scale == 'subset_250':
                            val_list_file = 'val_vggs_250.txt'
                        elif self.args.val_set_scale == 'subset_1k':
                            val_list_file = 'val_vggs_1k.txt'
                        else:
                            raise ValueError(
                                'VGGS validation supports val_set_scale '
                                'subset_250 or subset_1k, got {}'.format(
                                    self.args.val_set_scale))

                        with open('metadata/' + val_list_file, 'r') as f:
                            txt_reader = f.readlines()
                            for item in txt_reader:
                                data.append(item.rstrip('\n'))
                            self.audio_path = args.trainset_path + '/total_video_3s_audio/'
                            self.video_path = args.trainset_path + '/total_video_frames/'
                            self.sample_layout = 'vggs_train'

                    elif self.args.val_set == 'VGGSS':
                        if self.args.val_set_scale == 'subset_250':
                            val_list_file = 'test_vggss_250.txt'
                        else:
                            val_list_file = 'test_vggss_4911.txt'

                        with open('metadata/' + val_list_file, 'r') as f:
                            txt_reader = f.readlines()
                            for item in txt_reader:
                                data.append(item.split('.')[0])
                            self.audio_path = args.vggss_test_path + '/audio/'
                            self.video_path = args.vggss_test_path + '/frame/'
                            self.has_annotations = True
                            self.annotation_type = 'vggss'
                            self.sample_layout = 'flat_image'

                    elif self.args.val_set == 'SoundNet':
                        with open('metadata/test_flick.csv') as f:
                            csv_reader = csv.reader(f)
                            for item in csv_reader:
                                data.append(item[0])
                        
                            self.audio_path = args.soundnet_test_path + '/audio/'
                            self.video_path = args.soundnet_test_path + '/frame/'
                            self.has_annotations = True
                            self.annotation_type = 'soundnet'
                            self.sample_layout = 'flat_image'
                    else:
                        raise ValueError('Unknown validation set: {}'.format(
                            self.args.val_set))
           
        self.imgSize = args.image_size 

        self.AmplitudeToDB = audio_T.AmplitudeToDB()
        self.whisper_sample_rate = 16000
        self.beats_sample_rate = 16000
        self.whisper_feature_extractor = None
        if self.args.aud_backbone_type == 'whisper':
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.args.whisper_model_name)

        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        # Retrieve list of audio and video files
        self.video_files = []
   
        for item in data[:]:
            # Define audio path
            audio_check_path = os.path.join(self.audio_path, item + '.wav')
            
            # Define image path based on dataset and mode
            if self.sample_layout == 'vggs_train':
                image_check_path = os.path.join(self.video_path, item, '125.jpg')
            else:
                image_check_path = os.path.join(self.video_path, item + '.jpg')
            
            # Ensure both exist before appending
            if os.path.exists(audio_check_path) and os.path.exists(image_check_path):
                self.video_files.append(item)

        print("{0} requested dataset size: {1}".format(self.mode.upper() , len(data)))
        print("{0} actual available size: {1}".format(self.mode.upper() , len(self.video_files)))
        
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        if self.mode == 'train':

            if self.args.img_aug == 'moco_v1':
                augmentation = [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]

                self.img_transform = transforms.Compose(augmentation)

            elif self.args.img_aug == 'moco_v2':
                augmentation = [
                    transforms.RandomResizedCrop(224, scale=(0.3, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]

                self.img_transform = transforms.Compose(augmentation)
            
            else:
                print("[INFO] No image augmentation selected.")
                self.img_transform = transforms.Compose([
                    transforms.Resize(256, Image.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])    
                
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])            

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _apply_audaug_if_enabled(self, audio_features, time_mask_param, freq_mask_param):
        if (self.args.aud_aug=='SpecAug') and (self.mode=='train') and (random.random() < 0.8):
            if self.args.aud_backbone_type == 'beats':
                maskings = audio_T.TimeMasking(time_mask_param=time_mask_param)
            else:
                maskings = nn.Sequential(
                    audio_T.TimeMasking(time_mask_param=time_mask_param),
                    audio_T.FrequencyMasking(freq_mask_param=freq_mask_param)
                    )
            audio_features = maskings(audio_features)

        return audio_features

    def _build_audio_input(self, samples, samplerate):
        if self.args.aud_backbone_type == 'whisper':
            if samples.shape[0] > 1:
                samples = torch.mean(samples, dim=0, keepdim=True)
            if samplerate != self.whisper_sample_rate:
                samples = torchaudio.functional.resample(
                    samples, samplerate, self.whisper_sample_rate)
            input_features = self.whisper_feature_extractor(
                samples.squeeze(0).numpy(),
                sampling_rate=self.whisper_sample_rate,
                return_tensors='pt'
            ).input_features
            input_features = input_features.squeeze(0)
            input_features = self._apply_audaug_if_enabled(
                input_features, time_mask_param=180, freq_mask_param=11)
            return input_features

        if self.args.aud_backbone_type == 'beats':
            if samples.shape[0] > 1:
                samples = torch.mean(samples, dim=0, keepdim=True)
            if samplerate != self.beats_sample_rate:
                samples = torchaudio.functional.resample(
                    samples, samplerate, self.beats_sample_rate)
            samples = self._apply_audaug_if_enabled(
                samples, time_mask_param=self.beats_sample_rate, freq_mask_param=None)
            return samples.squeeze(0)

        spectrogram = audio_T.MelSpectrogram(
                sample_rate=samplerate,
                n_fft=512,
                hop_length=239,
                n_mels=257,
                normalized=True
            )(samples)

        spectrogram = self._apply_audaug_if_enabled(
            spectrogram, time_mask_param=180, freq_mask_param=35)

        return self.AmplitudeToDB(spectrogram)

    def __len__(self):
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]

        if self.sample_layout == 'vggs_train':
            frame = self.img_transform(self._load_frame(os.path.join( self.video_path, file , '125.jpg' ) ))
            samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))
            if samples.shape[0] > 1: # if stereo convert to mono
                samples = torch.mean(samples, dim=0, keepdim=True)

        else:
            frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '.jpg')  ))
            samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))
            if samples.shape[0] > 1: # if stereo convert to mono
                samples = torch.mean(samples, dim=0, keepdim=True)


        target_duration = 30 if self.args.aud_backbone_type == 'whisper' else 10
        target_num_samples = samplerate * target_duration

        if samples.shape[1] < target_num_samples:
            n = int(target_num_samples / samples.shape[1]) + 1
            samples = samples.repeat(1, n)

        samples = samples[...,:target_num_samples]

        spectrogram = self._build_audio_input(samples, samplerate)

        return frame, spectrogram, 'samples', file
