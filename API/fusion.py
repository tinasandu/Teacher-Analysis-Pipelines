def create_seq_scores(in_scores, frames_per_seq):
    scores = torch.tensor(in_scores) 
    if scores.shape[0] % frames_per_seq > 0:
      scores = F.pad(scores, (0, (frames_per_seq- scores.shape[0] % frames_per_seq)),  "constant", 0. )
    scores = scores.reshape(scores.shape[0] // frames_per_seq, frames_per_seq)
    return scores

def eval_fusion(input_seq_len, rating_seq_len, face_model, pose_model, voice_model, fusion_model):

  outs = []

  show_individual_graphs = True
  score=0.
  score_w=0.
  ccc=0.
  
  face_model, f_model_type, VA_face = face_model 
  pose_model, p_model_type = pose_model
  model_name, v_model_type, VA_voice = voice_model

  print("eval", input_seq_len)
  audio_scores, audio_ratings = gen_voice_graph_and_score(model_name, v_model_type, VA=VA_voice, show_graph=show_individual_graphs)
  face_scores, face_ratings = gen_face_graph_and_score(face_model, model_type=f_model_type, VA=VA_face, frames_per_seq=input_seq_len, show_graph=show_individual_graphs)
  pose_scores, pose_ratings = gen_pose_graph_and_score(pose_model, model_type=p_model_type, VA=False, frames_per_seq=input_seq_len, show_graph=show_individual_graphs)

  test_overall_ratings = torch.tensor(np.load(f"{content_path}/opinions/overall_ratings_labels_test_slice.npy"))
  test_overall_ratings_dev = torch.tensor(np.load(f"{content_path}/opinions/overall_ratings_devs_test_slice.npy"))

  plt.figure(figsize=(12, 8))
  test_overall_ratings = create_seq_scores(test_overall_ratings, input_seq_len)
  test_overall_ratings = torch.mean(test_overall_ratings, dim=1)

  test_overall_ratings_dev = create_seq_scores(test_overall_ratings_dev, input_seq_len)
  test_overall_ratings_dev = torch.mean(test_overall_ratings_dev, dim=1)

  face_ratings = create_seq_scores(face_ratings, rating_seq_len)
  pose_ratings = create_seq_scores(pose_ratings, rating_seq_len)

  audio_ratings = create_seq_scores(audio_ratings, rating_seq_len)

  print(audio_ratings.shape)
  print(face_ratings.shape)
  print(test_overall_ratings.shape)

  model = torch.load(f"{content_path}/models/{fusion_model}")
  model.eval()

  for seq in range(min(face_ratings.shape[0], audio_ratings.shape[0])): 
    # fusion
    f=face_ratings[seq]
    p=pose_ratings[seq]
    v=audio_ratings[seq]

    input = torch.stack([f, p, v], dim=1)
    # print(input.shape)
    # seq_dim = input.shape[0]
    input_dim = len(input[0].flatten())
    seq_len = input.shape[0]
    feature = Variable(input.view(-1, seq_len, input_dim).to(device))

    network_prediction = model(feature.float()).squeeze(-1)
    outs.append(network_prediction)

    score += torch.sum((network_prediction - test_overall_ratings[seq])**2)
    score_w += torch.sum(((network_prediction - test_overall_ratings[seq])**2 * (1/(1 + test_overall_ratings_dev[seq]))))
  
  outs = torch.stack(outs, 0)
  outs = outs.flatten().detach().cpu()
  test_overall_ratings = test_overall_ratings.flatten().cpu()

  plt.plot(range(outs.shape[0]), outs, label = 'Model Predictions')
  plt.plot(range(len(test_overall_ratings)), test_overall_ratings, label='True Mean Ratings')
  plt.plot(range(len(test_overall_ratings_dev)), test_overall_ratings_dev, label='Uncertainty factor')
  
  ccc += ccc_loss(outs, test_overall_ratings)

  plt.title("Engagement for Overall Performance predictions")
  # plt.xticks(range(0, outs.shape[0], 30), [i//6 for i in range(0, outs.shape[0], 30)])
  plt.xlabel("Time (seconds)")
  plt.ylabel("Predicted score")

  print()
  print(f'error for {fusion_model} on test set: {score.item()}')
  print(f'weighted error for {fusion_model} on test set: {score_w.item()}') 
  print()
  print(f'CCC score for {fusion_model} on test set: {ccc}')

  score_overall = 1.0 - torch.sum(torch.abs(test_overall_ratings) * (1/(1 + test_overall_ratings_dev))) / len(outs)
  
  print(f"overall score: {score_overall}")
  print('-----------------')
  plt.legend()
  plt.show() 


def train_model_fusion(network, name, loader_train, loader_val, criterion, 
                optimiser, single_cue_nets = None, device="cpu", 
                num_epochs=10, valid_freq=1, 
                VA=False, patience_threshold = 5,
                seq_len = 9, out_seq = 1):

      losses = []
      losses_validation = []
      iterations = []
      patience_threshold = patience_threshold
      patience = patience_threshold

      f, p, v = single_cue_nets
      path_f = f'{content_path}/models/{f}'
      path_p = f'{content_path}/models/{p}'
      path_v = f'{content_path}/models/{v}'

      scorer_f=torch.load(path_f)
      scorer_p=torch.load(path_p)
      scorer_v=torch.load(path_v)

      scorer_f.eval().to(device)
      scorer_p.eval().to(device)
      scorer_v.eval().to(device)

      netwrok=network.to(device)

      # aggregator = torch.nn.Linear(1, 9) 
      # aggregator = aggregator.to(device)

      for epoch in range(num_epochs):
        network.train()
        loss_train = 0
        train_total = 0
        ############################### Loop over training iterations  
        for iter, (features, o, o_std) in enumerate(loader_train):
          optimiser.zero_grad()
          o=o.flatten()
          o_std=o_std.flatten()

          std = Variable(o_std.to(device))
          o = Variable(o.to(device))
          o_std = o_std.to(device)

          with torch.no_grad():
            f = scorer_f(features[0].to(device))
            p = scorer_p(features[1].to(device)) 
            v = scorer_v(features[2].to(device))

          f = create_seq_scores(f.flatten(), 6) 
          p = create_seq_scores(p.flatten(), 6) 
          v = create_seq_scores(v.flatten(), 6) 
          # o = create_seq_scores(o, 6) 
          # o_std = create_seq_scores(o_std, 6) 

          input = torch.stack([f, p, v], dim=-1)
       
          # input_dim = len(input[0].flatten())
          # seq_len = input.shape[0]
          # print(input.view(1, seq_len, input_dim).shape)
          # feature = Variable(input.view(1, seq_len, input_dim).to(device))

          network_prediction = network(input.to(device))
          # print(network_prediction.shape)
          # print(o.shape)

          if out_seq == 1:
            # print(o.shape)
            # print(create_seq_scores(o, 54).shape)
            # aggregator.train()
            o = aggregator(create_seq_scores(o, 54).float())
            o_std = aggregator(create_seq_scores(o_std, 54).float())
          else:  
            o = create_seq_scores(o,9).float()
            o_std = create_seq_scores(o_std, 9).float()
            o = torch.mean(o, dim = -1)
            o_std = torch.mean(o_std, dim =-1)
            # o = o.reshape(9, 6)
            # o_std = o_std.reshape(9,6)
            o = create_seq_scores(o, seq_len).float()
            o_std = create_seq_scores(o_std, seq_len).float()

          if criterion == 'MSE':
            loss = torch.nn.MSELoss(reduction='sum')(network_prediction.float(), o.float())
          elif criterion == "CCC" and o.shape[0]>1:
            loss = ccc_loss(network_prediction.float(), o.float())
          else:
            loss = torch.sum(2/(1+ o_std) * (network_prediction - o) ** 2)

          loss.backward()
          loss_value = loss.item()

          train_total += o.size(0)

          if np.isnan(loss_value):
            print(loss)
            print(network_prediction)

          loss_train += loss_value

          optimiser.step()
          if iter % 20 == 0: 
            print('Epoch: %d, Iteration: %d, training loss = %.4f' % (epoch, iter, loss_value))         

        losses.append(loss_train/train_total)
        iterations.append(epoch)

        ###################### check performance on validation set
        if loader_val is not None and epoch % valid_freq == 0:
          network.eval()
          v_loss = 0
          v_total = 0
          with torch.no_grad():
            for v_iter, (v_features, vo, vo_std) in enumerate(loader_val):

              vo = vo.flatten().to(device)
              vo_std = vo_std.flatten().to(device)

              f = scorer_f(v_features[0].to(device))
              p = scorer_p(v_features[1].to(device)) 
              v = scorer_v(v_features[2].to(device))

              f = create_seq_scores(f.flatten(), 6) 
              p = create_seq_scores(p.flatten(), 6) 
              v = create_seq_scores(v.flatten(), 6) 
              # vo = create_seq_scores(vo.flatten(), 6) 
              # vo_std = create_seq_scores(vo_std.flatten(), 6) 

              input = torch.stack([f, p, v], dim=-1)

              # input_dim = len(input[0].flatten())
              # seq_len = input.shape[0]
              # feature = Variable(input.view(-1, seq_len, input_dim).to(device))

              v_outputs = network(input)
              
              # aggregator.eval()
              if out_seq == 1:
                vo = aggregator(create_seq_scores(vo, 54).float())
                vo_std = aggregator(create_seq_scores(vo_std, 54).float())
              else:
                vo = create_seq_scores(vo,9).float()
                vo_std = create_seq_scores(vo_std, 9).float()
                vo = torch.mean(vo, dim = -1)
                vo_std = torch.mean(vo_std, dim =-1)
                # vo = vo.reshape(9, 6)
                # vo_std = vo_std.reshape(9,6)
                vo = create_seq_scores(vo, seq_len).float()
                vo_std = create_seq_scores(vo_std, seq_len).float()  

              v_total += vo.size(0)
              vo=vo.to(device)
              vo_std = vo_std.to(device)

              # loss = criterion(v_outputs.float(), v_labels.float())
              if criterion == 'MSE':

                loss = torch.nn.MSELoss(reduction='sum')(v_outputs.float(), vo.float())
              elif criterion == "CCC" and vo.shape[0]>1:
                loss = ccc_loss(v_outputs, vo)
              else:
                loss = torch.sum(2/(1+ vo_std) * (v_outputs - vo) ** 2)

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

def train_fusion(network, name, loader_train, loader_val, criterion, 
                optimiser, device="cpu", num_epochs=10, valid_freq=1, 
                VA=False, patience_threshold = 5):


      network.to(device)

      losses = []
      losses_validation = []
      iterations = []
      patience_threshold = patience_threshold
      patience = patience_threshold

      for epoch in range(num_epochs):
        network.train()
        loss_train = 0
        train_total = 0
        ############################### Loop over training iterations  
        for iter, (features, o, o_std) in enumerate(loader_train):
          optimiser.zero_grad()

          f, p, v = features
          input = torch.stack([f, p, v], dim = -1)

          input_dim = input.shape[-1]
          seq_len = input.shape[1]

          feature = Variable(input.view(-1, seq_len, input_dim).to(device))
          # stds = (f_std + p_std + v_std)/3.
          std = Variable(o_std.to(device))
          o = Variable(o.to(device))
 
          network_prediction = network(feature.float()).squeeze(-1)
          
          if criterion == 'MSE':
            loss = torch.nn.MSELoss(reduction='sum')(network_prediction.float(), o.float())
          else:
            loss = torch.sum(2/(1+ std) * (network_prediction - o) ** 2)

          loss.backward()
          loss_value = loss.item()

          train_total += o.size(0)
          loss_train += loss_value

          optimiser.step()
          if epoch % 10 == 0 and iter % 20 == 0: 
            print('Epoch: %d, Iteration: %d, training loss = %.4f' % (epoch, iter, loss_value))         

        losses.append(loss_train/train_total)
        iterations.append(epoch)

        ###################### check performance on validation set
        if epoch % valid_freq == 0:
          network.eval()
          v_loss = 0
          v_total = 0
          with torch.no_grad():
            for v_iter, ((f,f_std), (p, p_std), (v, v_std), (vo, vo_std)) in enumerate(loader_val):

              input = torch.cat([f, p, v], dim=1)
              feature = Variable(input.view(-1, seq_len, input_dim).to(device))
              # stds = (f_std + p_std + v_std)/3.
              vo_std = Variable(vo_std.to(device))
              vo = Variable(vo.to(device))
              
              v_outputs = network(feature).squeeze(-1)
              # vo_std = Variable(vo_std.to(device))

              v_total += o.size(0)
              # loss = criterion(v_outputs.float(), v_labels.float())
              if criterion == 'MSE':
                loss = torch.nn.MSELoss(reduction='sum')(v_outputs.float(), o.float())
              else:
                loss = torch.sum(2/(1+ vo_std) * (v_outputs - vo) ** 2)

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
      # (name, feat, model_type = model_type, VA=VA, show_graph=True)

class PredictionDataset(Dataset):

    def __init__(self, dataset, seq_len=12, transform = True):

      frames_per_seq = seq_len
      self.features = []
      self.features_dev = []
      for i, f in enumerate(["face", "pose", "voice", "overall"]):
        self.features.append(Tensor(np.load(f"{content_path}/opinions/{f}_ratings_labels_{dataset}.npy")))
        self.features_dev.append(Tensor(np.load(f"{content_path}/opinions/{f}_ratings_devs_{dataset}.npy")))

        self.features[i] = F.pad(self.features[i], (0,(frames_per_seq- self.features[i].shape[0] % frames_per_seq)),  "constant", 0. )
        self.features[i] = self.features[i].reshape(self.features[i].shape[0]//frames_per_seq, frames_per_seq)
      
        self.features_dev[i] = F.pad(self.features_dev[i], (0,(frames_per_seq- self.features_dev[i].shape[0] % frames_per_seq)),  "constant", 0. )
        self.features_dev[i] = self.features_dev[i].reshape(self.features_dev[i].shape[0]//frames_per_seq, frames_per_seq)
          
    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, idx):
      face = self.features[0][idx]
      pose = self.features[1][idx]
      voice = self.features[2][idx]

      face_std = self.features_dev[0][idx]
      pose_std = self.features_dev[1][idx]
      voice_std = self.features_dev[2][idx]

      o = self.features[3][idx]
      o_std = self.features_dev[3][idx]

      return (face, face_std), (pose, pose_std), (voice, voice_std), (o, o_std)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def eval_fusion_att(seq_len, network_name, test_loader, out_len=6, 
                single_cue_nets=None, att=True, smoothen=False):

  outs = []
  test_ratings =[]
  test_ratings_dev =[]

  score_MSE=0
  score_wMSE=0
  score_CCC=0

  batch_size=1
 
  network = torch.load(f'{content_path}/models/{network_name}')
  network.eval()

  if single_cue_nets != None:  
    f, p, v = single_cue_nets

  else:
    f, p, v = ("BGRU_Att_face_align_seq_of_9", "GRU_Att_body_aligned_seq_9" ,"GRU_voice_va_short_seq")
  
  path_f = f'{content_path}/models/{f}'
  path_p = f'{content_path}/models/{p}'
  path_v = f'{content_path}/models/{v}'

  scorer_f=torch.load(path_f)
  scorer_p=torch.load(path_p)
  scorer_v=torch.load(path_v)

  scorer_f.eval().to(device)
  scorer_p.eval().to(device)
  scorer_v.eval().to(device)

  # if not att:
  #   aggregator = torch.load(f'{content_path}/models/{network_name}_agg')
  #   aggregator = aggregator.to(device)

  start_time = time.time()
  with torch.no_grad():
    for iter, (features, o, o_std) in enumerate(test_loader):
      if out_len==1:
        # o = torch.mean(o.flatten(), dim=-1) 
        # o_std = torch.mean(o_std.flatten(), dim=-1)
        o = torch.mean(o, dim=-1) 
        o_std = torch.mean(o_std, dim=-1)

      std = Variable(o_std.to(device))
      o = Variable(o.to(device))

      with torch.no_grad():
        if att:
          f, f_h = scorer_f(features[0].to(device), att)
          p, _= scorer_p(features[1].to(device), att) 
          v, _= scorer_v(features[2].to(device), att)

          network_prediction = network(f,p,v)

        else:  
          # print(features[0].shape)
          
          f = scorer_f(features[0].to(device), att)
          p = scorer_p(features[1].to(device), att) 
          v = scorer_v(features[2].to(device), att)
          
          # print(f.shape)

          f = create_seq_scores(f.flatten(), 6) 
          p = create_seq_scores(p.flatten(), 6) 
          v = create_seq_scores(v.flatten(), 6) 
          # print(f.shape)

          input = torch.stack([f, p, v], dim=-1)
          # print(input.shape)
       
          network_prediction = network(input.to(device))
          

          # if out_len == 1:
          #   continue
          #   o = create_seq_scores(o.flatten().to(device), 54).float()
          #   o_std = create_seq_scores(o_std.flatten().to(device), 54).float()

          if out_len == 6:  
            # o = create_seq_scores(o, 9).float()
            # o_std = create_seq_scores(o_std, 9).float()
            o = torch.mean(o, dim = -1)
            o_std = torch.mean(o_std, dim =-1)
            o = create_seq_scores(o, 6).float()
            o_std = create_seq_scores(o_std, 6).float()  

          # seq=6

          # f = f.unsqueeze(0)
          # p = p.unsqueeze(0)
          # v = v.unsqueeze(0)

          # o = torch.mean(o, dim=-1).unsqueeze(0)
          # o_std = torch.mean(o_std, dim=-1).unsqueeze(0)

          # input = torch.stack([f, p, v], dim=-1)
          # input_dim = 3
          # seq_len = input.shape[1]
          # feature = Variable(input.view(-1, seq_len, input_dim).to(device))
          # network_prediction = network(feature.float()).squeeze(-1)

          # print(network_prediction.shape)

      # print(network_prediction.squeeze(0).shape)
      outs.append(network_prediction.squeeze(0))
      test_ratings.append(o.squeeze(0))
      test_ratings_dev.append(o_std.squeeze(0))

  print(time.time()-start_time)
  
  outs= torch.cat(outs, dim=0)
  test_ratings = torch.cat(test_ratings, dim=0)
  test_ratings_dev = torch.cat(test_ratings_dev, dim=0)

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
  
  try:
    idxs = test_dataset.get_video_segmentation()

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

      print(f'''CCC scores for {network_name} fusion on test set 
    on each video: {scores_CCC_per_seq}''')

      print(f'''w_MSE scores for {network_name} fusion on test set 
    on each video: {scores_wMSE_per_seq}''')
  except:
    print("no vid segmentation")
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

  fusion_type = "Attention" if att else "Model"
  title = f"Multi-cue Predictions based on {fusion_type} Fusion vs True Overall Ratings"
  plt.title(title)
  print(len(outs))
  print(len(test_ratings_dev))

  # seq_len of 1.5 sec for now
  repeat = len(outs)//10
  plt.xticks(range(0, int(len(outs)), repeat), [int(i * 1.5) for i in range(0, len(outs), repeat)])

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

  eval_fusion_score_group(outs, test_ratings, title=title)

def ccc_loss(y_hat, y):
        a = concordance_cc2(y_hat.view(-1), y.view(-1), 'none')
        if a == 0: 
          return torch.tensor(1.)
        return 1 - a.squeeze()

def concordance_cc2_np(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)

def concordance_cc2(r1, r2, reduction='mean'):
    '''
    Computes batch sequence-wise CCC.
    '''
    # print(type(r1), type(r2))
    r1=r1.to(device)
    r2=r2.to(device)

    r1_mean = torch.mean(r1, dim=-1, keepdim=True)
    r2_mean = torch.mean(r2, dim=-1, keepdim=True)
    mean_cent_prod = torch.mean(((r1 - r1_mean) * (r2 - r2_mean)), dim=-1, keepdim=True)
    
    # print((r1.var(dim=-1, keepdim=True) + r2.var(dim=-1, keepdim=True) + (r1_mean - r2_mean) ** 2))
    # print((torch.var(r1, dim=-1, keepdim=True) + torch.var(r2, dim=-1, keepdim=True) + (r1_mean - r2_mean) ** 2))
    ccc = (2 * mean_cent_prod) / (r1.var(dim=-1, keepdim=True) + r2.var(dim=-1, keepdim=True) + (r1_mean - r2_mean) ** 2)
    if reduction == 'none':
        return ccc
    elif reduction == 'mean':

        return ccc.mean()  

class AttentionPredictionDataset(Dataset):

    def __init__(self, dataset, frames_per_seq=12, device=device):

      self.vectors = []

      self.audio_ratings = pd.read_csv(f'{content_path}/dataset/aud_3s/{dataset}/{dataset}_labels_stddev.csv')
      self.audio_dir = f'{content_path}/dataset/aud_3s/{dataset}'
      # print("audio", len(os.listdir(self.audio_dir))-2)

      for i, f in enumerate(["face", "pose"]):
        
        self.feat_dataset = torch.load(f"{content_path}/dataset/{f}_{dataset}_unfiltered")

        self.feat_dataset = F.pad(self.feat_dataset, (0, 0, 0, 0, 0, 0, 0, (frames_per_seq- self.feat_dataset.shape[0] % frames_per_seq)),  "constant", 0. )
        self.feat_dataset = self.feat_dataset.reshape(self.feat_dataset.shape[0]//frames_per_seq, frames_per_seq, self.feat_dataset.shape[1], self.feat_dataset.shape[2], self.feat_dataset.shape[3]) 
        self.vectors.append(self.feat_dataset)
  
      self.overall = (Tensor(np.load(f"{content_path}/opinions/overall_ratings_labels_{dataset}.npy")))
      self.overall_dev = (Tensor(np.load(f"{content_path}/opinions/overall_ratings_devs_{dataset}.npy")))

      self.overall = F.pad(self.overall, (0,(frames_per_seq- self.overall.shape[0] % frames_per_seq)),  "constant", 0. )
      self.overall = self.overall.reshape(self.overall.shape[0]//frames_per_seq, frames_per_seq)
      
      self.overall_dev = F.pad(self.overall_dev, (0,(frames_per_seq- self.overall_dev.shape[0] % frames_per_seq)),  "constant", 0. )
      self.overall_dev = self.overall_dev.reshape(self.overall_dev.shape[0]//frames_per_seq, frames_per_seq)

      print("faces:", self.vectors[0].shape)
      print("poses:", self.vectors[1].shape)
      print("voices:", len(self.audio_ratings)-1)

    def __len__(self):
      return min(len(self.overall), len(self.audio_ratings)-1)

    def __getitem__(self, idx):
      sample_file = f'{self.audio_dir}/{self.audio_ratings.iloc[idx, 0]}'
      clip = read(sample_file)
      audio_features = librosa.feature.mfcc(y=clip[1].astype(float), sr=clip[0], hop_length=512, n_fft=2048).transpose().astype(np.float32)

      self.feats= []
      for i,f in enumerate(["face","pose"]):
        vec = self.vectors[i][idx]
        frame_seq = align_crop_resize_per_seq(f, data=vec)
        frame_seq = frame_seq.flatten(start_dim=1)
        self.feats.append((frame_seq.float()))

      self.feats.append(torch.tensor(audio_features))  

      o = self.overall[idx]
      o_std = self.overall_dev[idx]

      return (self.feats, o, o_std)

class MultiModalDataset(Dataset):

    # def __init__(self, dataset, audio_path, audio_label_path,
    #                 video_path, video_label_path, 
    #                 frames_per_seq=12, device=device):

    def __init__(self, dataset, audio_path, audio_label_path,
                    video_path, video_label_path, 
                    frames_per_seq=12, device=device, out_seq=6, 
                    count_groups=False, model_type=None):


      VA_available=False   
      seq_len = frames_per_seq           

      if model_type == 'regression':
        self.voice_transform = False
      else:
        self.voice_transform = True

      if not count_groups:
        self.face_dataset = VisualFeatureDatasetLSTMfromDF("face", dataset, video_path, video_label_path, 
                                                VA=False, augment=False, transform = True, 
                                                align=self.voice_transform, frames_per_seq= seq_len, 
                                                device=device)

        self.pose_dataset =  VisualFeatureDatasetLSTMfromDF("pose", dataset, video_path, video_label_path, 
                                                VA=False, augment=False, transform = True, 
                                                align=self.voice_transform, frames_per_seq= seq_len, 
                                                device=device)

        self.voice_dataset = VocalSnippetsDatasetfromDF(path_labels=audio_label_path,path_audio=audio_path,  
                                                  dataset=dataset,
                                                  transform=self.voice_transform, VA=VA_available)

      # self.multi_dataset = torch.utils.data.ConcatDataset((self.face_dataset, self.pose_dataset, self.voice_dataset))                                          
                                         
      idx = []
      frames = []
      self.ratings = pd.read_csv(video_label_path)

      if 'eval' in dataset:
        vid = int(dataset[11])
        if dataset == f'eval_video_{vid}_whole':
          crit = self.ratings.index[self.ratings["video"]==vid]
        elif dataset == f'eval_video_{vid}_first_half':
          crit = self.ratings.index[self.ratings["video"]==vid]
          clips_per_vid = len(crit)
          crit = crit[:int(0.5*clips_per_vid)]
        elif dataset == f'eval_video_{vid}_second_half':
          crit = self.ratings.index[self.ratings["video"]==vid]
          clips_per_vid = len(crit)
          crit = crit[int(0.5*clips_per_vid):]

        idx.append(crit)

      elif "train_6" in dataset:
          for i in [1, 2, 3, 4, 5, 6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            # if dataset == "training":
            #   crit = crit[:int(0.7*clips_per_vid)]
            # elif dataset == "validation":
            #   crit = crit[int(0.7*clips_per_vid):int(0.9*clips_per_vid)]
            # elif dataset == "test":
            #   crit = crit[int(0.9*clips_per_vid):]-1

            # self.concatenation.append(self.concatenation[-1]+len(crit))
            idx.append(crit[:-1])
            

      elif "test_3" in dataset:
          for i in [2,4,6]:
            crit = self.ratings.index[self.ratings["video"]==i]
            clips_per_vid = len(crit)
            # self.concatenation.append(self.concatenation[-1]+len(crit))
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

      idx = np.concatenate(idx)
      
      self.ratings_mean = torch.tensor(self.ratings[f"mean_overall"][idx].values)
      self.ratings_std = torch.tensor(self.ratings[f"std_overall"][idx].values)

      self.ratings_mean = F.pad(self.ratings_mean, (0,(frames_per_seq- self.ratings_mean.shape[0] % frames_per_seq)),  "constant", 0. )
      self.ratings_mean = self.ratings_mean.reshape(self.ratings_mean.shape[0]//frames_per_seq, frames_per_seq)

      self.ratings_std = F.pad(self.ratings_std, (0,(frames_per_seq- self.ratings_std.shape[0] % frames_per_seq)),  "constant", 0. )
      self.ratings_std = self.ratings_std.reshape(self.ratings_std.shape[0]//frames_per_seq, frames_per_seq)
      
      if not out_seq:
        self.ratings_mean= torch.mean(self.ratings_mean, dim=1)
        self.ratings_std= torch.mean(self.ratings_std, dim=1)

      if count_groups:
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

#####################################
      # self.vectors = []

      # self.audio_ratings = pd.read_csv(f'{content_path}/dataset/aud_3s/{dataset}/{dataset}_labels_stddev.csv')
      # self.audio_dir = f'{content_path}/dataset/aud_3s/{dataset}'
      # # print("audio", len(os.listdir(self.audio_dir))-2)

      # for i, f in enumerate(["face", "pose"]):
        
      #   self.feat_dataset = torch.load(f"{content_path}/dataset/{f}_{dataset}_unfiltered")

      #   self.feat_dataset = F.pad(self.feat_dataset, (0, 0, 0, 0, 0, 0, 0, (frames_per_seq- self.feat_dataset.shape[0] % frames_per_seq)),  "constant", 0. )
      #   self.feat_dataset = self.feat_dataset.reshape(self.feat_dataset.shape[0]//frames_per_seq, frames_per_seq, self.feat_dataset.shape[1], self.feat_dataset.shape[2], self.feat_dataset.shape[3]) 
      #   self.vectors.append(self.feat_dataset)
  
      # self.overall = (Tensor(np.load(f"{content_path}/opinions/overall_ratings_labels_{dataset}.npy")))
      # self.overall_dev = (Tensor(np.load(f"{content_path}/opinions/overall_ratings_devs_{dataset}.npy")))

      # self.overall = F.pad(self.overall, (0,(frames_per_seq- self.overall.shape[0] % frames_per_seq)),  "constant", 0. )
      # self.overall = self.overall.reshape(self.overall.shape[0]//frames_per_seq, frames_per_seq)
      
      # self.overall_dev = F.pad(self.overall_dev, (0,(frames_per_seq- self.overall_dev.shape[0] % frames_per_seq)),  "constant", 0. )
      # self.overall_dev = self.overall_dev.reshape(self.overall_dev.shape[0]//frames_per_seq, frames_per_seq)

      # print("faces:", self.vectors[0].shape)
      # print("poses:", self.vectors[1].shape)
      # print("voices:", len(self.audio_ratings)-1)

    def __len__(self):
      # return min(len(self.ratings_mean), len(self.audio_ratings)-1)
      return min(len(self.face_dataset), len(self.voice_dataset)-1)

    def get_video_segmentation(self):
      return self.face_dataset.get_video_segmentation()

    def __getitem__(self, idx):
      
      v, v_l, _ = self.voice_dataset.__getitem__(idx)
      f, f_l, _ = self.face_dataset.__getitem__(idx)
      p, p_l, _ = self.pose_dataset.__getitem__(idx)

      # m = self.multi_dataset.__get_item__(idx)
      # print(m)
      try:
        o, o_std = self.ratings_mean[idx], self.ratings_std[idx]
      except:
        print(idx)
        print(self.ratings_mean)
      # sample_file = f'{self.audio_dir}/{self.audio_ratings.iloc[idx, 0]}'
      # clip = read(sample_file)
      # audio_features = librosa.feature.mfcc(y=clip[1].astype(float), sr=clip[0], hop_length=512, n_fft=2048).transpose().astype(np.float32)

      # self.feats= []
      # for i,f in enumerate(["face","pose"]):
      #   vec = self.vectors[i][idx]
      #   frame_seq = align_crop_resize_per_seq(f, data=vec)
      #   frame_seq = frame_seq.flatten(start_dim=1)
      #   self.feats.append((frame_seq.float()))

      # self.feats.append(torch.tensor(audio_features))  

      # o = self.overall[idx]
      # o_std = self.overall_dev[idx]

      if self.voice_transform:
        v=torch.tensor(v)
        if v.shape[0] != 259: 
          # print(type(v))
          v = F.pad(v, (0,0,0, 259 - v.shape[0]), "constant", 0)

      # print(f.shape)
      # print(v)
      # print(o.shape)
      return ([f, p, v], o, o_std)
      # return o, o_std
      

def train_fusion_with_att(name, loader_train, loader_val, criterion, device="cpu",
                          single_cue_nets=None, hidden_dim=64,
                          num_epochs=10, learning_rate=0.0001, valid_freq=1,
                          VA=False, patience_threshold = 5, out_seq=9, att_ver=1):
      losses = []
      losses_validation = []
      iterations = []
      patience_threshold = patience_threshold
      patience = patience_threshold

      f, p, v = single_cue_nets
      path_f = f'{content_path}/models/{f}'
      path_p = f'{content_path}/models/{p}'
      path_v = f'{content_path}/models/{v}'

      scorer_f=torch.load(path_f)
      scorer_p=torch.load(path_p)
      scorer_v=torch.load(path_v)

      scorer_f.eval().to(device)
      scorer_p.eval().to(device)
      scorer_v.eval().to(device)
      if att_ver == 1:
        network = AttentionFusion([64,50,128], out_dim=out_seq)
      else:
        network = AttentionFusion2([64,50,128], hidden_dim, out_dim=out_seq)
      optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
      network.to(device)

      for epoch in range(num_epochs):
        network.train()
        loss_train = 0
        train_total = 0
        ############################### Loop over training iterations  
        for iter, (features, o, o_std) in enumerate(loader_train):
          optimiser.zero_grad()

          std = Variable(o_std.to(device))
          o = Variable(o.to(device))
          o_std=o_std.to(device)

          with torch.no_grad():
            f, f_h = scorer_f(features[0].to(device), True)
            p, _= scorer_p(features[1].to(device), True) 
            v, _= scorer_v(features[2].to(device), True)

          network_prediction = network(f,p,v)
          # print('...................')
          # print(o.shape)
          # print(network_prediction.shape)

          if out_seq == 1:
            o = torch.mean(o, dim=-1)
            o_std = torch.mean(o_std, dim=-1)

          if criterion == 'MSE':
            loss = torch.nn.MSELoss(reduction='sum')(network_prediction.float(), o.float())
          elif criterion == "CCC" and o.shape[0]>1:
            loss = ccc_loss(network_prediction.float(), o.float())
          else:
            loss = torch.sum(2/(1+ o_std) * (network_prediction - o) ** 2)

          loss.backward()
          loss_value = loss.item()

          train_total += o.size(0)

          if np.isnan(loss_value):
            print(loss)
            print(network_prediction)

          loss_train += loss_value

          optimiser.step()
          if iter % 20 == 0: 
            print('Epoch: %d, Iteration: %d, training loss = %.4f' % (epoch, iter, loss_value))         

        losses.append(loss_train/train_total)
        iterations.append(epoch)

        ###################### check performance on validation set
        if epoch % valid_freq == 0:
          network.eval()
          v_loss = 0
          v_total = 0
          with torch.no_grad():
            for v_iter, (v_features, vo, vo_std) in enumerate(loader_val):

              f, _= scorer_f(v_features[0].to(device), True)
              p, _= scorer_p(v_features[1].to(device), True) 
              v, _= scorer_v(v_features[2].to(device), True)

              v_outputs = network(f,p,v)
              
              if out_seq ==1:
                vo = torch.mean(vo, dim=-1)
                vo_std = torch.mean(vo_std, dim=-1)

              v_total += vo.size(0)
              vo=vo.to(device)
              vo_std = vo_std.to(device)

              # loss = criterion(v_outputs.float(), v_labels.float())
              if criterion == 'MSE':

                loss = torch.nn.MSELoss(reduction='sum')(v_outputs.float(), vo.float())
              elif criterion == "CCC" and vo.shape[0]>1:
                loss = ccc_loss(v_outputs, vo)
              else:
                loss = torch.sum(2/(1+ vo_std) * (v_outputs - vo) ** 2)

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
      # (name, feat, model_type = model_type, VA=VA, show_graph=True)

# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#         self.v = nn.Parameter(torch.rand(hidden_size))
#         stdv = 1. / math.sqrt(self.v.size(0))
#         self.v.data.uniform_(-stdv, stdv)

#     def forward(self, hidden, encoder_outputs):
#         timestep = encoder_outputs.size(1)
#         h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # B*T*H
#         # encoder_outputs: [B*T*H]
#         attn_energies = self.score(h, encoder_outputs)
#         return F.softmax(attn_energies, dim=1).unsqueeze(1)

#     def score(self, hidden, encoder_outputs):
#         # [B*T*2H]->[B*T*H]
#         energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
#         energy = energy.transpose(1, 2)  # [B*H*T]
#         v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
#         energy = torch.bmm(v, energy)  # [B*1*T]
#         return energy.squeeze(1)  # [B*T]

class AttFusion(nn.Module):
    def __init__(self, input_dim=[512, 512, 512]):
        super(AttFusion, self).__init__()
        hidden_dim = 64
        seq_len = 9
        
        self.scorer_f = GRUModel_Att(input_dim[2], hidden_dim, 1, seq_len, device=device, bidir=True)
        self.scorer_p = GRUModel_Att(input_dim[2], hidden_dim, 1, seq_len, device=device, bidir=True)
        self.scorer_v = GRUModel_Att(input_dim[2], hidden_dim, 1, seq_len, device=device, bidir=True)

        self.projection = torch.nn.Linear(259, 9)
        
        self.scorer_f.train()
        self.scorer_p.train()
        self.scorer_v.train()

    def forward(self, x_f, x_p, x_v):
        x_f = F.pad(x_f, (0, (x_v.shape[-1]-x_f.shape[-1])), "constant", 0.)
        x_p = F.pad(x_p, (0, (x_v.shape[-1]-x_p.shape[-1])), "constant", 0.)
        
        x_v = self.projection(x_v.reshape(x_v.shape[0], x_v.shape[2], x_v.shape[1]))
        x_v = x_v.reshape(x_v.shape[0], x_v.shape[2], x_v.shape[1])
        
        # x_f = 64, 9, 64

        f = self.scorer_f(x_f.to(device))
        p = self.scorer_p(x_p.to(device))
        v = self.scorer_v(x_v.to(device))

        h_f = torch.sigmoid(f.unsqueeze(-1))
        h_p = torch.sigmoid(p.unsqueeze(-1))
        h_v = torch.sigmoid(v.unsqueeze(-1))

        # 64, 259, 1
        # 64, 259, 1
        # 64, 259, 1 
        
        h = torch.cat((h_f, h_p, h_v), dim=-1) # (B, T, 3)
        h = F.softmax(h, dim=-1)

        f = h[..., 2].unsqueeze(-1) * x_v +  h[..., 0].unsqueeze(-1)* x_f+  h[..., 1].unsqueeze(-1) * x_p
        # print(f.shape)

        return f

# class AttentionFusion(nn.Module):
#   def __init__(self, input_dims, out_dim=9, version=1):
#     super(AttentionFusion, self).__init__()
#     if version == 1:
#       self.att_fuse = AttFusion(input_dims)
#     elif version == 2:
#       self.
#     self.att_fuse.to(device).train()
#     self.hidden = 64
#     self.fcs = torch.nn.Sequential( nn.Linear(2*self.hidden, self.hidden),
#                                     nn.ReLU(True),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(self.hidden, out_dim)
#                                    )

class AttentionFusion(nn.Module):
  def __init__(self, input_dims, out_dim=9 ):
    super(AttentionFusion, self).__init__()
    self.att_fuse = AttFusion(input_dims)
    self.att_fuse.to(device).train()
    self.hidden = 64
    self.fcs = torch.nn.Sequential( nn.Linear(2*self.hidden, self.hidden),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(self.hidden, out_dim)
                                   )

    self.fusion = GRUModel_Att(128, self.hidden, layer_dim=1, output_dim=out_dim, device=device, bidir = True)
    self.fusion.train()
    
  def forward(self, face, pose, voice):
    features = self.att_fuse(face, pose, voice)
    # print(features.shape)
    # return self.fusion(features.unsqueeze(1))
    fused,_ = self.fusion(features, True)
    # print(fused.shape)
    res = self.fcs(fused[:,-1,:])
    return res

class AttFusion2(nn.Module):
    def __init__(self, input_dim=[512, 512, 512], hidden_dim=64):
        super(AttFusion2, self).__init__()
        hidden_dim = hidden_dim
        seq_len = 9
        num_heads = 4

        self.encoding_f = torch.nn.Linear(input_dim[0], hidden_dim, device=device)
        self.encoding_p = torch.nn.Linear(input_dim[1], hidden_dim, device=device)
        self.encoding_v = torch.nn.Linear(input_dim[2], hidden_dim, device=device)

        self.projection = torch.nn.Linear(259, 9)

        # embed dim is hidden dim x3 as we cat the encodings of face, pose and voice
        self.attention = torch.nn.MultiheadAttention(hidden_dim * 3, num_heads, dropout=0.2, bias=True, batch_first=True, device=device)

    def forward(self, x_f, x_p, x_v):
        # x_f = F.pad(x_f, (0, (x_v.shape[-1]-x_f.shape[-1])), "constant", 0.)
        # x_p = F.pad(x_p, (0, (x_v.shape[-1]-x_p.shape[-1])), "constant", 0.)
        
        # x_v = B, 259, 128
        # TODO: experiment with doing projection before and after linear layer
        x_v = self.projection(x_v.reshape(x_v.shape[0], x_v.shape[2], x_v.shape[1]))
        x_v = x_v.reshape(x_v.shape[0], x_v.shape[2], x_v.shape[1])
        
        # x_f = B, 9, 64
        # x_p = B, 9, 50
        # x_v = B, 9, 128

        f_enc = self.encoding_f(x_f.to(device))
        p_enc = self.encoding_p(x_p.to(device))
        v_enc = self.encoding_v(x_v.to(device))

        # Each encoding should be (batch_size, seq_len, hidden_dim)        

        h_cat = torch.cat((f_enc, p_enc, v_enc), dim=-1)

        h, _ = self.attention(h_cat, h_cat, h_cat)
        return h

# TODO: Sort out hidden dims for attn and Grus
class AttentionFusion2(nn.Module):
  def __init__(self, input_dims, hidden_dim=64, out_dim=1):
    super(AttentionFusion2, self).__init__()
    self.att_fuse = AttFusion2(input_dims, hidden_dim)
    self.att_fuse.to(device).train()
    self.hidden = hidden_dim
    self.fusion = GRUModel_Att(self.hidden * 3, self.hidden, layer_dim=2, output_dim=1, device=device, bidir = True)
    self.fusion.train()
    
  def forward(self, face, pose, voice):
    attn_features = self.att_fuse(face, pose, voice)
    # print(attn_features.shape)
    fused = self.fusion(attn_features)
    # print(fused.shape)
    return fused