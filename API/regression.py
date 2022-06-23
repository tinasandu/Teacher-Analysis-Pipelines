def create_dataloader(feat, name, VA=False, align=False):
      print(device)
      face_dataset_v = torch.load(f"{content_path}/dataset/{feat}s_videos_stack_{name}")
      genders_v = torch.load(f"{content_path}/dataset/genders_{name}")
      face_ratings_v = np.load(f"{content_path}/opinions/{feat}_ratings_labels_{name}.npy")
      face_ratings_dev_v = np.load(f"{content_path}/opinions/{feat}_ratings_devs_{name}.npy")

      if align:
        for i in range(face_dataset_v.shape[0]):
          face = align_crop_resize_per_frame(feature="face", data_feature=face_dataset_v[i])
          face_dataset_v[i] = F.pad(face, (0,1), "constant", 0.)  
      if VA:
        va = torch.load(f"{content_path}/opinions/{feat}_va_{name}")
        vas = []
        for i in range(6):
          va_i = torch.stack(va[i])
          vas.append(va_i)
        va = torch.cat(vas, 0)  
        
        genders_v[:,0,0,1:] = va 

      mask = face_dataset_v.cpu()[:, 0, 0 ,0] > -99.0
      face_dataset_v = face_dataset_v[mask]
      face_ratings_v = face_ratings_v[mask]
      face_ratings_dev_v = face_ratings_dev_v[mask]
      if VA:
        va = va[mask]
      genders_v = genders_v[mask].float()

      print(f'{name} set shape is {face_dataset_v.shape}')

      return TensorDataset(torch.tensor(genders_v.float(), device=device),
                          torch.tensor(face_dataset_v, device=device),
                          torch.tensor(face_ratings_v, device=device),
                          torch.tensor(face_ratings_dev_v, device=device))

# # Create a Network class, which inherits the torch.nn.Module class, which represents a neural network.
class Network_Face(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input.
    def __init__(self, input_dimension, output_dimension):
        super(Network_Face, self).__init__()
        print(input_dimension, output_dimension)

        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.LeakyReLU()
        )
        self.output_layer = torch.nn.Linear(in_features=512, out_features=output_dimension)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer_2(x)
        x = self.output_layer(x)
        return x

def create_trainable_Network_Face(input_dimension):
    # Create the neural network
    network = Network_Face(input_dimension=input_dimension, output_dimension=1)

    return network  

# Create a Network class, which inherits the torch.nn.Module class, which represents a neural network.
class Network_Body(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input.
    def __init__(self, input_dimension, output_dimension):
        super(Network_Body, self).__init__()
        print(input_dimension, output_dimension)

        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=256, out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.LeakyReLU()
        )
        self.output_layer = torch.nn.Linear(in_features=512, out_features=output_dimension)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer_2(x)
        x = self.output_layer(x)
        return x

def create_trainable_Network_Body(input_dimension):
    # Create the neural network
    network = Network_Body(input_dimension=input_dimension, output_dimension=1)

    return network  

# Main entry point
def train_for_dataset(network, name, feat, VA=False, lr=0.0001, num_epochs=10, align=False):

      # Create the optimiser
      optimiser = torch.optim.Adam(network.parameters(), lr=lr)    

      losses = []
      losses_validation = []
      iterations = []

      fig, ax = plt.subplots()
      ax.set(xlabel='Iteration', ylabel='Loss', title=f'Loss Curve for {name}')

      # n = len(dataset)
      # n_val = int(n / 10)

      batch_size = 64

      ## train set
      dataset_train = create_dataloader(feat, 'training_slice_va', VA, align=align)
  
      ## validation set
      
      dataset_validate =  create_dataloader(feat, 'validation_slice_va', VA, align=align)

      ## Dataloaders and training

      loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
      loader_val = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

      patience_threshold = 7
      patience = patience_threshold

      network.to(device)
      for training_iteration in range(num_epochs):
        optimiser.zero_grad()

        ############################### Loop over training iterations  
        network.train()
        loss_train = 0
        for t, (g, x, y, y_dev) in enumerate(loader_train):
            
            input = torch.cat((g.float(), x.float()), 2).flatten(start_dim=1)

            network_prediction = network.forward(input).squeeze(1)
            # loss = torch.nn.MSELoss()(network_prediction, y.float()) * y_dev

            loss = torch.sum(2/(1+y_dev) * (network_prediction - y.float()) ** 2)
            
            loss.backward()
            optimiser.step()
            loss_value = loss.item()
            if training_iteration%10 == 0 and t % 100 == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (training_iteration, t, loss_value))

            loss_train += loss_value

        losses.append(loss_train/(t+1))
        iterations.append(training_iteration)

        ###################### check performance on validation set
        network.eval()
        loss_sum = 0
        with torch.no_grad():
            for t, (g, x, y, y_dev) in enumerate(loader_val):
                input = torch.cat((g.float(), x.float()), 2).flatten(start_dim=1)

                network_prediction = network.forward(input).squeeze(1)

                # using weighted loss
                loss = torch.sum(2/(1+y_dev) * (network_prediction - y.float()) ** 2)
                
                if training_iteration%10 == 0 and t == 5:
                  print('Epoch: %d, Iteration %d validation, loss = %.4f' % (training_iteration, t, loss.item()))

                loss_sum += loss.item()

        losses_validation.append(loss_sum/(t+1))

        if training_iteration > 10 and (losses_validation[training_iteration - 1] < losses_validation[training_iteration]):
            patience -= 1
        else:
            patience = patience_threshold

        if patience == 0:
            break

      # Plot and save the loss vs iterations graph
      plt.ylim([0, 300])
      ax.plot(range(len(losses[1:])), losses[1:], color='blue')
      ax.plot(range(len(losses_validation[1:])), losses_validation[1:], color='orange')

      plt.show()
      fig.savefig("loss_vs_iterations.png")

      torch.save(network, f"{content_path}/models/{name}")

def eval_hand_crafted_fusion():
  # import matplotlib.pyplot as plt
  # from importlib import reload
  # plt = reload(plt)

  test_dataset_face = torch.load(f"{content_path}/dataset/faces_videos_stack_test_slice").cpu()
  test_dataset_pose = torch.load(f"{content_path}/dataset/poses_videos_stack_test_slice").cpu()

  test_genders = torch.load(f"{content_path}/dataset/genders_test_slice")
  test_face_ratings = np.load(f"{content_path}/opinions/face_ratings_labels_test_slice.npy")
  test_pose_ratings = np.load(f"{content_path}/opinions/pose_ratings_labels_test_slice.npy")
  test_overall_ratings = np.load(f"{content_path}/opinions/overall_ratings_labels_test_slice.npy")

  test_overall_ratings_dev = np.load(f"{content_path}/opinions/overall_ratings_devs_test_slice.npy")

  model_face = torch.load(f"{content_path}/models/face4_VA")
  model_pose = torch.load(f"{content_path}/models/body3")
  # model_voice = torch.load(f"{content_path}/models/voice1")

  # audio processing
  audios = []
  path = f'{content_path}/dataset/aud/test/'

  for j, filename in enumerate(os.listdir(path)):
    if(filename[-3:] == "wav"):
      v, l = aT.file_regression(f'{path}/{filename}', f"{content_path}/models/voice_rdmforrest", "randomforest")
      # print(v, l, )
      audios.append(v[0])

  print('no of video clips', len(audios))

  audio_models = np.repeat(audios, 3*3)
  print(len(audios))
  print(test_dataset_face.shape) 
  pad = np.full((test_dataset_face.shape[0] - len(audio_models),), 0.0)
  audio_models = np.concatenate([pad, audio_models])

  outs=[]
  for f in range(test_dataset_face.shape[0]):
    out_face = -5
    out_pose = -5

    model_face.eval()
    if not torch.all(test_dataset_face[f] == 0) and torch.all(test_dataset_face[f] > -99.0):
      with torch.no_grad():
        input = torch.cat((test_genders[f].float(), test_dataset_face[f].float()), 1).flatten(start_dim=1)
        out_face = model_face.forward(input).item()

    model_pose.eval()
    if not torch.all(test_dataset_pose[f] == 0) and torch.all(test_dataset_pose[f] > -99.0):
      with torch.no_grad():
        input = torch.cat((test_genders[f].float(), test_dataset_pose[f].float()), 1).flatten(start_dim=1)
        out_pose = model_pose.forward(input).item()

    if out_face>-5 and out_pose > -5:
      # this formula obtained 0.71
      outs.append((2*out_face + out_pose + audio_models[f])/5 - 0.5 )

      # this obtains 0.65
      # outs.append((2*out_face + out_pose + audio_models[f])/4 - 1)
    elif out_face>-5:
      outs.append((2*out_face+audio_models[f])/4)
    elif out_pose > -5:
      outs.append((out_pose + audio_models[f])/4)
    else:
      outs.append(0)  

  outs = np.array(outs)
  print(f' Test sequence consists of {len(test_overall_ratings)/6} seconds')

  score = 1.0 - (np.sum(np.abs(np.array(outs) - test_overall_ratings) * (1/(1+test_overall_ratings_dev)))) / len(outs)
  print(f"Score for the current model: {score}")

  score = 1.0 - (np.sum(np.abs(np.array(p) - test_overall_ratings) * (1/(1+test_overall_ratings_dev)))) / len(p)
  print(f"Score for the current model after smoothening: {score}")

  ccc = ccc_loss(torch.tensor(p), torch.tensor(test_overall_ratings))
  print(f"CCC for the current model after smoothening: {ccc}")

  plt.figure(figsize=(12, 8))
  plt.title(f'Network prediction vs measured results ')


  def moving_average(x, w):
      return np.convolve(x, np.ones(w), 'valid') / w
      
  seq=20
  p = outs[:-12].reshape(len(outs)//seq,seq)
  p = np.mean(p, axis=1)
  p = np.repeat(p, seq)
  # p = np.cumsum(p) / [1+i for i in range(len(p))]
  from scipy.signal import savgol_filter

  p = savgol_filter(outs, 101, 3)

  plt.plot(range(len(p)), p, color='blue', alpha=0.8, linewidth=1.5, label='Network Prediction')
  # plt.plot(range(len(outs)), np.cumsum(outs)/[1+i for i in range(len(outs))], color='blue', alpha=0.8, linewidth=1.0, label='network prediction')
  plt.plot(range(len(test_overall_ratings_dev)), test_overall_ratings_dev, color='red', linewidth=2.0, label = 'Uncertainty Factor')
  plt.plot(range(len(test_overall_ratings)), test_overall_ratings, color='orange', linewidth=3.0, label= 'Real Engagement Ratings')

  # plt.text(2, 3, f'Score obtained for overall engagement: {score}',
  #      horizontalalignment='center',
  #      verticalalignment='center')

  plt.xlabel('Time')
  plt.ylabel('Overall Engagement')
  x = list(range(10))
  plt.xticks(range(0, len(p), 60), [i//6 for i in range(0, len(p), 60)])

  # plt.xticks(list(range(len(outs))), range(len(outs)//6)) 
  plt.legend()
  plt.show()

execfile(f'{content_path}/API/helpers.py')
def eval_fusion_regression(seq_len, test_loader,
                single_cue_nets=None, smoothen=False):

  outs = []
  test_ratings =[]
  test_ratings_dev =[]

  score_MSE=0
  score_wMSE=0
  score_CCC=0

  batch_size=1
 
  # network = torch.load(f'{content_path}/models/{network_name}')
  # network.eval()

  if single_cue_nets != None:  
    f, p, v = single_cue_nets

  else:
    f, p, v = ("BGRU_Att_face_align_seq_of_9", "GRU_Att_body_aligned_seq_9" ,"GRU_voice_va_short_seq")
  
  path_f = f'{content_path}/models/{f}'
  path_p = f'{content_path}/models/{p}'
  # path_v = f'{content_path}/models/{v}'

  scorer_f=torch.load(path_f)
  scorer_p=torch.load(path_p)
  # scorer_v=torch.load(path_v)

  scorer_f.eval().to(device)
  scorer_p.eval().to(device)
  # scorer_v.eval().to(device)

  test_genders = torch.load(f"{content_path}/dataset/genders_test_slice")

  start_time = time.time()
  with torch.no_grad():
    for iter, (features, o, o_std) in enumerate(test_loader):

      f_all, p_all, v_all = features 
      voices=[]

      for i in range(len(v_all)):
        v_, l = aT.file_regression(v_all[i], f"{content_path}/models/voice_rdmforrest", "randomforest")
        voices.append(v_[0])

      outs_ = []
      out_faces=[]
      out_poses=[]

      out_voice=voices[0]

      for i in range(f_all.shape[1]):
        out_face = -5
        out_pose = -5
        f = f_all[:,i,:]
        p = p_all[:,i,:]

        gender = torch.zeros((f.shape[0],3)).cuda()
        
        if not torch.all(f == 0) and torch.all(f > -99.0):
          with torch.no_grad():
            
            input = torch.cat((gender.float(), f.float()), -1).flatten(start_dim=1)
            out_face = scorer_f(input).item()
        out_faces.append(out_face)

        if not torch.all(p == 0) and torch.all(p > -99.0):
          with torch.no_grad():
            input = torch.cat((gender.float(), p.float()), -1).flatten(start_dim=1)
            out_pose = scorer_p(input).item()
        out_poses.append(out_pose)

      out_pose = np.mean(out_poses)
      out_face = np.mean(out_faces)

      if out_face>-5 and out_pose > -5:
        # this formula obtained 0.71
        outs.append((2*out_face + out_pose + out_voice)/5 - 0.5 )

        # this obtains 0.65
        # outs.append((2*out_face + out_pose + audio_models[f])/4 - 1)
      elif out_face>-5:
        outs.append((2*out_face + out_voice)/4)
      elif out_pose > -5:
        outs.append((out_pose + out_voice)/4)
      else:
        outs.append(0)  
          # print(network_prediction.squeeze(0).shape)
      # print(out_face)
      
      # outs.append((torch.tensor(outs_)).float())
      # print("here")
      # print("outs:", torch.tensor(outs_).shape)
      # print("o:", o.shape)
      test_ratings.append(torch.mean(o.squeeze(0)).flatten())
      test_ratings_dev.append(torch.mean(o_std.squeeze(0)).flatten())

  print(len(outs))
  outs= torch.tensor(outs)
  print(outs.shape)


  test_ratings = torch.cat(test_ratings)
  test_ratings_dev = torch.cat(test_ratings_dev)

  test_ratings_dev=test_ratings_dev.to(device)
  test_ratings = test_ratings.to(device)
  outs = outs.to(device)

  # if (outs.shape[1] == 1):
  #   print("here")
  #   print(test_ratings.shape)
  #   test_sratings = test_ratings.mean(axis=1)
  #   test_ratings_dev = test_ratings_dev.mean(axis=1)

  test_ratings = test_ratings.flatten()  
  test_ratings_dev = test_ratings_dev.flatten()  

  outs=outs.flatten()


  # outs = torch.tensor(moving_average(outs.cpu(), 50))
  # with torch.no_grad():
  #   outs = torch.nn.Conv1d(1, 1, 11, padding=5, bias=False, padding_mode='zeros', device=device)(outs.unsqueeze(0))
  #   outs = outs.squeeze()

  if smoothen:
    outs = torch.tensor(savgol_filter(np.array(outs.cpu()), 21, 3))
  outs = outs.to(device)

  print('here')
  print(test_ratings.shape)
  print(outs.shape)

  loss = torch.nn.MSELoss(reduction='sum')(outs.float(), test_ratings.float())
  score_MSE+= loss.item() 
  loss = concordance_cc2(outs, test_ratings)
  score_CCC+=loss.item()
  loss = torch.sum(2/(1+ test_ratings_dev) * (outs - test_ratings) ** 2)      
  score_wMSE+=loss.item()   
                                    
  plt.figure(figsize=(12, 8))

   ################
  # sequence concatenation
  ################
  scores_CCC_per_seq = []
  scores_wMSE_per_seq = []
  
  idxs = test_dataset.get_video_segmentation()
  network_name = 'hand-crafted'    


  print("here")
  print(idxs)
  if len(idxs) > 0:
    for i in range(len(idxs[:-1])):
      fr = idxs[i]//9
      next_fr = idxs[i+1]//9

      plt.axvline(x=fr, linewidth = 2, alpha = 0.2, color = 'green')
      loss_per_vid = concordance_cc2(outs[fr:next_fr], test_ratings[fr:next_fr])
      scores_CCC_per_seq.append(loss_per_vid.item())
      loss_per_vid = torch.nn.MSELoss(reduction='sum')(outs[fr:next_fr], test_ratings[fr:next_fr])
      sc = score = 1.0 - (torch.sum(torch.abs(outs[fr:next_fr] - \
        test_ratings[fr:next_fr]) * (1/(1+test_ratings_dev[fr:next_fr])))) \
        / len(outs[fr:next_fr])
      scores_wMSE_per_seq.append(sc.item())


    print(f'''CCC scores for hand-crafted fusion on test set 
  on each video: {scores_CCC_per_seq}''')

    print(f'''w_MSE scores for hand-crafted fusion on test set 
  on each video: {scores_wMSE_per_seq}''')
  ##################    

  print()
  # test_ratings = torch.stack(test_ratings).flatten()
  score = 1.0 - (torch.sum(torch.abs(outs - test_ratings) * (1/(1+test_ratings_dev)))) / len(outs)
  print(f"Weigthed score for {network_name} for fusion with attention after smoothening: {score}")
  print(f'CCC score for {network_name} for fusion with attention on test set: {score_CCC}')
  print()

  plt.plot(range(outs.shape[0]), outs.cpu(), label = 'Model Predictions', marker='*')
  plt.plot(range(len(test_ratings)), test_ratings.cpu(), label='True Ratings', marker = '.')

  # plt.plot(range(len(test_ratings_dev)), test_ratings_dev.cpu(), label='Uncertainty Factor')
  test_ratings_dev = test_ratings_dev.cpu()
  test_ratings = test_ratings.cpu()

  plt.fill_between(range(len(test_ratings_dev)), test_ratings+test_ratings_dev/2, test_ratings-test_ratings_dev/2, facecolor='r', alpha=0.2)
  
  print()
  print(f'MSE error for {network_name} on test set: {score_MSE}')
  print(f'weighted MSE error for {network_name} on test set: {score_wMSE}')

  plt.title(f"Multi-cue Predictions based on Attention Fusion vs True Overall Ratings")
  print(len(outs))
  print(len(test_ratings_dev))

  # seq_len of 1.5 sec for now
  plt.xticks(range(0, int(len(outs)), 10), [int(i * 1.5) for i in range(0, len(outs), 10)])

  plt.xlabel('Time (seconds)')
  plt.ylabel('Overall Engagement')

  # print()
  # # test_ratings = torch.stack(test_ratings).flatten()
  # score = 1.0 - (torch.sum(torch.abs(outs - test_ratings) * (1/(1+test_ratings_dev)))) / len(outs)
  # print(f"Weigthed score for the current model after smoothening: {score}")
  # print(f'CCC score for {network_name} on test set: {score_CCC}')

  # plt.plot(range(outs.shape[0]), outs.cpu(), label = 'Model Predictions')
  # plt.plot(range(len(test_ratings)), test_ratings.cpu(), label='True Ratings')
  # plt.plot(range(len(test_ratings_dev)), test_ratings_dev.cpu(), label='Uncertainty Factor')
  # print()
  # print(f'MSE error for {network_name} on test set: {score_MSE}')
  # print(f'weighted MSE error for {network_name} on test set: {score_wMSE}')

  # score_overall = 1.0 - torch.sum(torch.abs(test_overall_ratings) * (1/(1 + test_overall_ratings_dev))) / len(outs)
  
  # print(f"overall score: {score_overall}")
  # print('-----------------')

  plt.legend()
  plt.show() 

  eval_fusion_score_group(outs, test_ratings)  