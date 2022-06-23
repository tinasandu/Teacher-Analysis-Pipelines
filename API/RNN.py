import os
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import math
import cv2
import os
from scipy.signal import savgol_filter

execfile('/content/drive/MyDrive/thesis_pipeline/API/helpers.py')

def train_for_feat_datasetLSTM(network, name, loader_train, loader_val, criterion, optimiser, device="cpu", num_epochs=10, valid_freq=1, VA=False, patience_threshold = 5):
      (name, feat, model_type) = name
      VA, va_padding = VA
      network.to(device)

      losses = []
      losses_validation = []
      iterations = []
      patience = patience_threshold

      for epoch in range(num_epochs):
        network.train()
        loss_train = 0
        train_total = 0
        ############################### Loop over training iterations  
        for iter, (input, labels, stds) in enumerate(loader_train):
          # print(input)
          optimiser.zero_grad()
          seq_dim = input.shape[1]
          input_dim = input.shape[-1]
          
          feature = Variable(input.view(-1, seq_dim, input_dim)).to(device)
          labels = Variable(labels.to(device))
          stds = Variable(stds.to(device))
                    
          network_prediction = network(feature.float()).squeeze(-1)
          # print(network_prediction)
          # print(labels)
          
          if criterion == 'MSE':
            loss = torch.nn.MSELoss(reduction='sum')(network_prediction.float(), labels.float())
          else:
            loss = torch.sum(2/(1+ stds) * (network_prediction - labels) ** 2)

          loss.backward()
          loss_value = loss.item()

          train_total += labels.size(0)
          loss_train += loss_value

          optimiser.step()
          if epoch % 10 == 0 and iter % 10 == 0: 
            print('Epoch: %d, Iteration: %d, training loss = %.4f' % (epoch, iter, loss_value))         

        losses.append(loss_train/train_total)
        iterations.append(epoch)

        ###################### check performance on validation set
        if epoch % valid_freq == 0:
          network.eval()
          v_loss = 0
          v_total = 0
          with torch.no_grad():
            for v_iter, (input, v_labels, v_stds) in enumerate(loader_val):
              
              v_feat = Variable(input.view(-1, seq_dim, input_dim)).to(device)

              v_labels = Variable(v_labels.to(device))

              v_outputs = network(v_feat).squeeze(-1)
              v_stds = Variable(v_stds.to(device))

              v_total += v_labels.size(0)
              # loss = criterion(v_outputs.float(), v_labels.float())
              if criterion == 'MSE':
                loss = torch.nn.MSELoss(reduction='sum')(v_outputs.float(), v_labels.float())
              else:
                loss = torch.sum(2/(1+ v_stds) * (v_outputs - v_labels) ** 2)

              v_loss_value = loss.item()
              v_loss += v_loss_value

              if iter % 20 == 0: 
                print('Epoch: %d, Iteration: %d, valid loss = %.4f' % (epoch, v_iter, v_loss_value))         
              
          losses_validation.append(v_loss/v_total)

        if epoch > 10 and (losses_validation[int(epoch / valid_freq) - 1] < losses_validation[int(epoch / valid_freq)]):
            patience -= 1
        else:
            patience = patience_threshold

        if patience == 0:
            break

      plot_loss_iters(name, content_path, valid_freq, losses, losses_validation)
      torch.save(network, f"{content_path}/models/{name}")

class VisualFeatureDatasetLSTMfromDF(Dataset):

    def __init__(self, feat, dataset, path_frames="", path_labels="", VA=True, augment=False, transform=False, align=True, frames_per_seq=12, device = 'cuda:0', out_seq=False):
        self.feat = feat
        self.dataset = dataset
        self.align = align
        self.transform = transform
        self.device = device
        self.VA = VA

        self.ratings = pd.read_csv(path_labels)
        # print(len(os.listdir(path_frames)))

        idx = []
        frames = []
        self.concatenation = [0]
        crit=[]

        if 'eval' in dataset:
          vid = int(dataset[11])
          self.concatenation.append(len(crit))
          if dataset == f'eval_video_{vid}_whole':
            crit = self.ratings.index[self.ratings["video"]==vid]
            # crit = crit[:720]
          elif dataset == f'eval_video_{vid}_first_half':
            crit = self.ratings.index[self.ratings["video"]==vid]
            clips_per_vid = len(crit)
            crit = crit[:int(0.5*clips_per_vid)]
          elif dataset == f'eval_video_{vid}_second_half':
            crit = self.ratings.index[self.ratings["video"]==vid]
            clips_per_vid = len(crit)
            crit = crit[int(0.5*clips_per_vid):]-1

          idx.append(crit)
          feature_vec = torch.load(f'{path_frames}{vid}/features/{feat}_features')

          selection = (feature_vec[self.ratings.loc[crit, "frame_number"].values])
          frames.append(selection)
          
        elif "train_6" in dataset:
          for i in [1,2, 3, 4, 5, 6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            # if dataset == "training":
            #   crit = crit[:int(0.7*clips_per_vid)]
            # elif dataset == "validation":
            #   crit = crit[int(0.7*clips_per_vid):int(0.9*clips_per_vid)]
            # elif dataset == "test":
            #   crit = crit[int(0.9*clips_per_vid):]-1

            crit = crit[:-1]
            self.concatenation.append(self.concatenation[-1]+len(crit))
  
            idx.append(crit)
            feature_vec = torch.load(f'{path_frames}{i}/features/{feat}_features')

            selection = (feature_vec[self.ratings.loc[crit, "frame_number"].values])
            frames.append(selection)

        elif "test_3" in dataset:
          for i in [6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            crit = crit[:-1]
            self.concatenation.append(self.concatenation[-1]+len(crit))
            idx.append(crit)
            
            feature_vec = torch.load(f'{path_frames}{i}/features/{feat}_features')

            selection = (feature_vec[self.ratings.loc[crit, "frame_number"].values])
            frames.append(selection)     

        else:
          for i in [ 1,2,3, 4, 5, 6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            if dataset == "training":
              crit = crit[:int(0.7*clips_per_vid)]
            elif dataset == "validation":
              crit = crit[int(0.7*clips_per_vid):int(0.9*clips_per_vid)]
            elif dataset == "test":
              crit = crit[int(0.9*clips_per_vid):]-1

            self.concatenation.append(self.concatenation[-1]+len(crit))
  
            idx.append(crit)
            feature_vec = torch.load(f'{path_frames}{i}/features/{feat}_features')

            selection = (feature_vec[self.ratings.loc[crit, "frame_number"].values])
            frames.append(selection)

        idx = np.concatenate(idx)

#########################
        # feature_vec = torch.load(f"{content_path}/dataset/faces_videos_stack_test_slice")
        # self.feat_dataset = feature_vec
#########################
        self.feat_dataset = torch.cat(frames)

        self.idxs = self.ratings.index[idx].tolist()
        self.video_nos = self.ratings.loc[idx, "video"].values
        
        self.ratings_mean = torch.tensor(self.ratings[f"mean_{feat}"][idx].values)
        self.ratings_std = torch.tensor(self.ratings[f"std_{feat}"][idx].values)
        print()
        print("load from dataframe - raw frames")
        print(self.ratings_mean.shape)
        print(self.ratings_std.shape)
        print(self.feat_dataset.shape)

###################
        # self.ratings_mean = torch.tensor(np.load(f"{content_path}/opinions/face_ratings_labels_test_unfiltered.npy"))
        # self.ratings_std =  torch.tensor(np.load(f"{content_path}/opinions/face_ratings_devs_test_unfiltered.npy"))
######################

        # if VA:
        #   self.va = torch.load(f"{content_path}/opinions/{self.feat}_va_best_{self.dataset}")

        # ratings = pd.read_csv(f'{content_path}/test_pipeline/opinions.csv')
        # idx = ratings.index[ratings["video"] == 2].tolist()
        # print(idx)
        # frames = ratings["frame_number"][idx].values
        # print(frames)
        # print(len(frames))

        # self.feat_dataset = torch.load(f'{content_path}/test_pipeline/videos_eval/video_b2_2/features/{feat}_features')
        # print(self.feat_dataset.shape)
        # self.feat_dataset = self.feat_dataset[frames]

        # else:  
        #   self.feat_dataset = torch.load(f"{content_path}/dataset/{self.feat}_{dataset}")
        #   # self.genders = torch.load(f"{content_path}/dataset/genders_{dataset}")
        #   self.ratings = np.load(f"{content_path}/opinions/{self.feat}_ratings_labels_{dataset}.npy")
        #   self.ratings_std = np.load(f"{content_path}/opinions/{self.feat}_ratings_devs_{dataset}.npy")
        #   self.ratings = Tensor(self.ratings)
        #   self.ratings_std = Tensor(self.ratings_std)

        if transform:
          self.ratings_mean = F.pad(self.ratings_mean, (0,(frames_per_seq- self.ratings_mean.shape[0] % frames_per_seq)),  "constant", 0. )
          self.ratings_mean = self.ratings_mean.reshape(self.ratings_mean.shape[0]//frames_per_seq, frames_per_seq)

          self.ratings_std = F.pad(self.ratings_std, (0,(frames_per_seq- self.ratings_std.shape[0] % frames_per_seq)),  "constant", 0. )
          self.ratings_std = self.ratings_std.reshape(self.ratings_std.shape[0]//frames_per_seq, frames_per_seq)
          
          if not out_seq:
            self.ratings_mean= torch.mean(self.ratings_mean, dim=1)
            self.ratings_std= torch.mean(self.ratings_std, dim=1)

          self.feat_dataset = F.pad(self.feat_dataset, (0, 0, 0, 0, 0, 0, 0, (frames_per_seq- self.feat_dataset.shape[0] % frames_per_seq)),  "constant", 0. )
          self.feat_dataset = self.feat_dataset.reshape(self.feat_dataset.shape[0]//frames_per_seq, frames_per_seq, self.feat_dataset.shape[1], self.feat_dataset.shape[2], self.feat_dataset.shape[3]) 

          if self.align:
            print(self.feat_dataset.shape)
            # print(time.time())
            self.feat_dataset = [align_crop_resize_per_seq(self.feat, data=face_frame, img_size=self.ratings["image_size"][idx[i]]) for i, face_frame in enumerate(self.feat_dataset)]
            self.feat_dataset = torch.stack(self.feat_dataset, dim=0)
            # print(time.time())

          if VA:
            self.va = torch.cat(self.va, 0)
            self.va = F.pad(self.va, (0, 1 ,0,(frames_per_seq- self.va.shape[0] % frames_per_seq)),  "constant", 0. )
            self.va = self.va.reshape(self.va.shape[0]//frames_per_seq, frames_per_seq, 1, 1, 3)

          if augment:
            good = self.ratings_mean > 2.5
            good = good.nonzero()
            n_good = torch.logical_and(self.ratings_mean > 0, self.ratings_mean <= 2.5)
            n_good = n_good.nonzero()
            n_bad = torch.logical_and(self.ratings_mean > -2.5, self.ratings_mean <= 0)
            n_bad = n_bad.nonzero()
            bad = self.ratings_mean <= -2.5
            bad=bad.nonzero()

            print('good', len(good))
            print('n_good', len(n_good))
            print('n_bad', len(n_bad))
            print('bad', len(bad))

            augs_good = 3
            augs_n_bad = 1
            
            res_aug = []
            res_std_aug =[]

            feature_vec = self.feat_dataset
            feat_dataset_aug = torch.zeros((augs_good*len(good) + augs_n_bad*len(n_bad) + feature_vec.shape[0],
               feature_vec.shape[1], feature_vec.shape[2], feature_vec.shape[3], feature_vec.shape[4]))
            feat_dataset_aug[:feature_vec.shape[0]] = self.feat_dataset

            for i in range(len(good)):
              aug = augment_dataset(self.feat_dataset[good[i]], augs_good).flatten(start_dim=0, end_dim=1)
              feat_dataset_aug[feature_vec.shape[0] + i*augs_good: feature_vec.shape[0] + (i+1)*augs_good]
              res_aug.append(self.ratings_mean[good[i]].repeat_interleave(augs_good, dim=0))
              res_std_aug.append(self.ratings_std[good[i]].repeat_interleave(augs_good, dim=0))

            for i in range(len(n_bad)):
              aug = augment_dataset(self.feat_dataset[n_bad[i]], augs_n_bad).flatten(start_dim=0, end_dim=1)
              feat_dataset_aug[feature_vec.shape[0] + i*augs_n_bad: feature_vec.shape[0] + (i+1)*augs_n_bad]
              res_aug.append(self.ratings_mean[n_bad[i]].repeat_interleave(augs_n_bad, dim=0))
              res_std_aug.append(self.ratings_std[n_bad[i]].repeat_interleave(augs_n_bad, dim=0))  

            self.ratings_mean = torch.cat([self.ratings_mean, torch.cat(res_aug)])
            self.ratings_std = torch.cat([self.ratings_std, torch.cat(res_std_aug)])  
            self.feat_dataset = feat_dataset_aug

          print()
          print(f"here - number of sequences for {feat}")
          print(self.feat_dataset.shape)
          # print(self.feat_dataset[-1])
          # print(len([p for p in range(len(self.feat_dataset)) if torch.all(self.feat_dataset) == 0.]))
          # print(len([p for p in range(len(self.feat_dataset)) if torch.all(self.feat_dataset) == 200.]))
          # print([i for i in range(self.feat_dataset[0]) if torch.all(self.feat_dataset[i]) == 0.])
          self.non_frames = torch.all(self.feat_dataset.flatten(start_dim = 1) == 0, dim = 1)
          # print(p.shape)
          # p = torch.tensor([torch.logical_and(p[i].flatten()) for i in range(p.shape[0])])
          # print(p)
          # data = data[np.array(data).sum(axis=1) != 0.]
          # data = data[np.array(data).sum(axis=1) != -200.])
          
          print(self.ratings_mean.shape)

    def __len__(self):
        return len(self.ratings_mean)

    def get_non_available_frames(self):
      return self.non_frames    
      
    def get_video_segmentation(self):
      return self.concatenation

    def __getitem__(self, idx):

        # gender = self.genders[idx]
        face_frame = self.feat_dataset[idx].to(device)
        rating = self.ratings_mean[idx]
        rating_std = self.ratings_std[idx]

        if self.transform:
          if self.VA:
            va = self.va[idx].to(device)
            frame_seq = torch.cat((va, frame_seq), 2)

          # frame_seq = frame_seq.flatten(start_dim=1)
        features = face_frame.flatten(start_dim=1).float()

        return features, rating, rating_std

class VisualFeatureDatasetLSTM(Dataset):

    def __init__(self, feat, dataset, path=None, VA=True, transform=False, align=True, frames_per_seq=12, device = 'cuda:0', out_seq=False):
        self.feat = feat
        self.dataset = dataset
        self.align = align
        self.transform = transform
        self.device = device
        self.VA = VA

        if VA:
          self.va = torch.load(f"{content_path}/opinions/{self.feat}_va_best_{self.dataset}")

        if path is not None:
          ratings = ""
        self.feat_dataset = torch.load(f"{content_path}/dataset/{self.feat}_{dataset}")
        # self.genders = torch.load(f"{content_path}/dataset/genders_{dataset}")
        self.ratings = np.load(f"{content_path}/opinions/{self.feat}_ratings_labels_{dataset}.npy")
        self.ratings_std = np.load(f"{content_path}/opinions/{self.feat}_ratings_devs_{dataset}.npy")
        self.ratings = Tensor(self.ratings)
        self.ratings_std = Tensor(self.ratings_std)

        if transform:
          self.ratings = F.pad(self.ratings, (0,(frames_per_seq- self.ratings.shape[0] % frames_per_seq)),  "constant", 0. )
          self.ratings = self.ratings.reshape(self.ratings.shape[0]//frames_per_seq, frames_per_seq)

          self.ratings_std = F.pad(self.ratings_std, (0,(frames_per_seq- self.ratings_std.shape[0] % frames_per_seq)),  "constant", 0. )
          self.ratings_std = self.ratings_std.reshape(self.ratings_std.shape[0]//frames_per_seq, frames_per_seq)
          
          if not out_seq:
            self.ratings= torch.mean(self.ratings, dim=1)
            self.ratings_std= torch.mean(self.ratings_std, dim=1)

          self.feat_dataset = F.pad(self.feat_dataset, (0, 0, 0, 0, 0, 0, 0, (frames_per_seq- self.feat_dataset.shape[0] % frames_per_seq)),  "constant", 0. )
          self.feat_dataset = self.feat_dataset.reshape(self.feat_dataset.shape[0]//frames_per_seq, frames_per_seq, self.feat_dataset.shape[1], self.feat_dataset.shape[2], self.feat_dataset.shape[3]) 

          if VA:
            self.va = torch.cat(self.va, 0)
            self.va = F.pad(self.va, (0, 1 ,0,(frames_per_seq- self.va.shape[0] % frames_per_seq)),  "constant", 0. )
            self.va = self.va.reshape(self.va.shape[0]//frames_per_seq, frames_per_seq, 1, 1, 3)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):

        # gender = self.genders[idx]
        face_frame = self.feat_dataset[idx].to(device)
        rating = self.ratings[idx]
        rating_std = self.ratings_std[idx]

        if self.transform:
          if self.align:
            frame_seq = align_crop_resize_per_seq(self.feat, data=face_frame)

          if self.VA:
            va = self.va[idx].to(device)
            frame_seq = torch.cat((va, frame_seq), 2)

          frame_seq = frame_seq.flatten(start_dim=1)
          features = frame_seq.float()

        return features, rating, rating_std

class GRUModel_Att(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device = "cpu", dropout_prob = 0.2, bidir=False):
        super(GRUModel_Att, self).__init__()
        self.device = device

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidir)

        # Fully connected layer
        if bidir:
          self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:  
          self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_h=False):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h = self.gru(x, h0.detach())

        # Convert the final state to our desired output shape (batch_size, output_dim)
        
        if return_h:
          return out, h

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device = "cpu", dropout_prob = 0.2, bidir=False):
        super(GRUModel, self).__init__()
        self.device = device

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidir)

        # Fully connected layer
        if bidir:
          self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:  
          self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out        

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device = "cpu", dropout_prob = 0.2, bidir=False):
        super(LSTMModel, self).__init__()
        self.device = device

        ## TODO: add in bidirectional parameter

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidir)

        # Fully connected layer
        if bidir:
          self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else: 
          self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.lstm(x)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
class RNNModel(nn.Module):
  def __init__(self, model_type, input_dim, hidden_dim, layer_dim, output_dim, device = "cpu", dropout_prob = 0.2, bidir=False):
    super(RNNModel, self).__init__()
    if model_type == "GRU":
      self.model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, device, dropout_prob, bidir)
    elif model_type == "LSTM":
      self.model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, device, dropout_prob, bidir)
    elif model_type == "GRU_Att":
      self.model = GRUModel_Att(input_dim, hidden_dim, layer_dim, output_dim, device, dropout_prob, bidir)

  def forward(self, x):
    return self.model(x)

import os
import pandas as pd

# class VisualFeatureDataset(Dataset):

#     def __init__(self, feat, dataset, VA=True, transform=False, align=True, device = 'cuda:0'):
#         self.feat = feat
#         self.dataset = dataset
#         self.align = align
#         self.transform = transform
#         self.device = device
#         self.VA = VA

#         if VA:
#           self.va = torch.load(f"{content_path}/opinions/{self.feat}_va_best_{self.dataset}")

#         # self.feat_dataset = torch.load(f"{content_path}/dataset/{self.feat}s_videos_stack_{dataset}")
#         self.feat_dataset = torch.load(f"{content_path}/dataset/{self.feat}_{dataset}")
#         self.genders = torch.load(f"{content_path}/dataset/genders_{dataset}")
#         self.ratings = np.load(f"{content_path}/opinions/{self.feat}_ratings_labels_{dataset}.npy")
#         self.ratings_std = np.load(f"{content_path}/opinions/{self.feat}_ratings_devs_{dataset}.npy")
#         self.ratings = Tensor(self.ratings)
#         self.ratings_std = Tensor(self.ratings_std)

#         if transform:
#           self.mask = self.feat_dataset[:, 0, 0 , 0] > -99.0
#           self.feat_dataset = self.feat_dataset[self.mask]
#           self.genders = self.genders[self.mask]
#           self.ratings = self.ratings[self.mask]
#           self.ratings_std = self.ratings_std[self.mask]

#           if VA:
#             self.va = torch.cat(self.va, 0)
#             self.va = self.va[self.mask]

#     def __len__(self):
#         return len(self.ratings)

#     def __getitem__(self, idx):

#         gender = self.genders[idx]
#         face_frame = self.feat_dataset[idx]

#         rating = self.ratings[idx]
#         rating_std = self.ratings_std[idx]

#         if self.transform:
#           if self.align:
#             face_frame = align_crop_resize_per_frame(self.feat, data_feature=face_frame)
#             face_frame = F.pad(face_frame, (0,1), "constant", 0.)
          
#           if self.VA:
#             self.genders[:,0,0,1:] = self.va 
#           features = torch.cat((gender.to(device), face_frame.to(device)), 1).flatten(start_dim=1)
#           features = features.float()

#         return features, rating, rating_std

def train_for_audio_dataset(network, name, loader_train, loader_val, criterion, optimiser, device="cpu", num_epochs=10, valid_freq=1, VA=False, patience_threshold = 5):
      network.to(device)

      losses = []
      losses_validation = []
      iterations = []
      patience = patience_threshold

      for epoch in range(num_epochs):
        network.train()
        loss_train = 0
        train_total = 0
        ############################### Loop over training iterations  
        for iter, (audio, labels, stds) in enumerate(loader_train):
          optimiser.zero_grad()
      
          seq_dim, input_dim = audio.shape[1], audio.shape[2]
          audio = Variable(audio.view(-1, seq_dim, input_dim).to(device))
          labels = Variable(labels.to(device))
          stds = Variable(stds.to(device))
 
          network_prediction = network(audio.float()).squeeze(-1)

          # loss = torch.nn.MSELoss()(network_prediction, y.float()) * y_dev
          # loss = torch.sum(2/(1+y_dev) * (network_prediction - y.float()) ** 2)
          if criterion == 'MSE':
            loss = torch.nn.MSELoss(reduction='sum')(network_prediction.float(), labels.float())
          elif criterion == "CCC":
            loss = ccc_loss(network_prediction, labels)
              
          else:
            loss = torch.sum(2/(1+ stds) * (network_prediction - labels) ** 2)

          loss.backward()
          loss_value = loss.item()

          train_total += labels.size(0)
          loss_train += loss_value

          optimiser.step()
          print('Epoch: %d, Iteration: %d, training loss = %.4f' % (epoch, iter, loss_value))         

        losses.append(loss_train/train_total)
        iterations.append(epoch)

        ###################### check performance on validation set
        if epoch % valid_freq == 0:
          network.eval()
          v_loss = 0
          v_total = 0
          with torch.no_grad():
            for v_iter, (v_audio, v_labels, v_stds) in enumerate(loader_val):
              v_audio = Variable(v_audio.view(-1, seq_dim, input_dim)).to(device)
              v_labels = Variable(v_labels.to(device))

              v_outputs = network(v_audio).squeeze(-1)
              v_stds = Variable(v_stds.to(device))

              v_total += v_labels.size(0)
              # loss = criterion(v_outputs.float(), v_labels.float())
              if criterion == 'MSE':
                loss = torch.nn.MSELoss(reduction='sum')(v_outputs.float(), v_labels.float())
              elif criterion == "CCC":
                loss = ccc_loss(v_outputs, v_labels)
              else:
                loss = torch.sum(2/(1+ v_stds) * (v_outputs - v_labels) ** 2)

              v_loss_value = loss.item()
              v_loss += v_loss_value
              print('Epoch: %d, Iteration: %d, valid loss = %.4f' % (epoch, v_iter, v_loss_value))         
              
          losses_validation.append(v_loss/v_total)

          if epoch > 10 and (losses_validation[int(epoch / valid_freq) - 1] < losses_validation[int(epoch / valid_freq)]):
              patience -= 1
          else:
              patience = patience_threshold

          if patience == 0:
              break

      plot_loss_iters(name, content_path, valid_freq, losses, losses_validation)

      torch.save(network, f"{content_path}/models/{name}") 

class VocalSnippetsDatasetfromDF(Dataset):

    def __init__(self, path_labels, path_audio, dataset, transform=False, VA=False, count_groups=False):
        self.dataset = dataset
        self.transform = transform
        self.VA = VA
        self.ratings = pd.read_csv(path_labels)
        print(len(os.listdir(path_audio)))
        self.path_audio = path_audio

        idx = []
        self.concatenation = [0]
        crit=[]

        if 'eval' in dataset:
          vid = int(dataset[11])
          self.concatenation.append(len(crit))
          if dataset == f'eval_video_{vid}_whole':
            crit = self.ratings.index[self.ratings["video"]==vid]
            # crit = crit[:80]
          elif dataset == f'eval_video_{vid}_first_half':
            crit = self.ratings.index[self.ratings["video"]==vid]
            clips_per_vid = len(crit)
            crit = crit[:int(0.5*clips_per_vid)]
          elif dataset == f'eval_video_{vid}_second_half':
            crit = self.ratings.index[self.ratings["video"]==vid]
            clips_per_vid = len(crit)
            crit = crit[int(0.5*clips_per_vid):]-1

          idx.append(crit)

        elif "train_6" in dataset:
          for i in [1, 2, 3, 4, 5, 6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            self.concatenation.append(self.concatenation[-1]+len(crit))
  
            idx.append(crit[:-1])

        elif "test_3" in dataset:
          for i in [6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            self.concatenation.append(self.concatenation[-1]+len(crit))
            idx.append(crit[:-1])

        else:    
          for i in [1,2,3,4,5,6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            if dataset == "training":
              crit = crit[:int(0.7*clips_per_vid)]
            elif dataset == "validation":
              crit = crit[int(0.7*clips_per_vid):int(0.9*clips_per_vid)]
            elif dataset == "test":
              crit = crit[int(0.9*clips_per_vid):]-1
  
            idx.append(crit)

            self.concatenation.append(self.concatenation[-1]+len(crit))

        idx = np.concatenate(idx)
        self.idxs = self.ratings.index[idx].tolist()
        # clips = self.audio_ratings["audio_clip"][idxs].values

        # self.audio_dir = torch.tensor(os.listdir(path_audio))[idx]
        # self.ratings = torch.tensor(ratings[f"mean"][idx].values)
        # self.ratings_std = torch.tensor(ratings[f"std"][idx].values)

        print()
        print(f"here - number of audio sequences")
        print(len(self.idxs))

        if count_groups:
          mean_ratings = torch.tensor(self.ratings["mean"].values)
          good = torch.tensor(mean_ratings > 2.5)
          good = good.nonzero()
          n_good = torch.logical_and(mean_ratings > 0, mean_ratings <= 2.5)
          n_good = n_good.nonzero()
          n_bad = torch.logical_and(mean_ratings > -2.5, mean_ratings <= 0)
          n_bad = n_bad.nonzero()
          bad = mean_ratings <= -2.5
          bad=bad.nonzero()

          print('good', len(good))
          print('n_good', len(n_good))
          print('n_bad', len(n_bad))
          print('bad', len(bad))

        # start_time = time.time()
        # for i in self.idxs:
        #   idx = self.idxs[i]
        #   sample_file = f"{self.path_audio}/{self.ratings['audio_clip'][idx]}"
        #   clip = read(sample_file)
        #   features = librosa.feature.mfcc(y=clip[1].astype(float), sr=clip[0], hop_length=512, n_fft=2048).transpose().astype(np.float32)

        # print(time.time() - start_time)    

    def __len__(self):
        return len(self.idxs)

    def get_video_segmentation(self):
      return self.concatenation    

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        sample_file = f"{self.path_audio}/{self.ratings['audio_clip'][idx]}"

        label = self.ratings['mean'][idx]
        std = self.ratings['std'][idx]
        # print(std)

        if not self.transform:
          # clip = read(sample_file)
          return sample_file, label, std
          
        else:
            # librosa
            clip = read(sample_file)
            features = librosa.feature.mfcc(y=clip[1].astype(float), sr=clip[0], hop_length=512, n_fft=2048).transpose().astype(np.float32)
            
            # pyAudio
            # [Fs, x] = audioBasicIO.read_audio_file(sample_file)
            # F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs, deltas=False)
            # # print(f_names)
            # # mfcc features for now
            # # features = F[0,1,2, 8:20, :]
            # features = F[8:20, :]

            if self.VA:
              va, l = aT.file_regression(sample_file, f"{content_path}/models/va_voice_svm", "svm")
              p1d = (0, features[1] - len(va))
              va = F.pad(torch.Tensor(va).view(1,2), (0, 18), "constant", 0)
              features = torch.cat((va, torch.tensor(features)), dim=0)
              return features, label, std
            else:
              return features, label, std