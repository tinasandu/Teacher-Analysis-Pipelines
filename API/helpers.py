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

def align_crop_resize_per_frame(feature, img_size = None, data_feature=None, path=None, show_ref_images = False, img_path=None):
  if feature == "face":
    POIS = [42, 39]
    ideal_dist = 100.

  elif feature == "pose":
    POIS = [2, 5]
    ideal_dist = 200.
    
  else:
    return

  if img_size is None:
    h, w = (360, 640)
  else:
    if type(img_size) is str:
      img_size = img_size[1:-1].split(',')

    h = int(img_size[0])
    w = int(img_size[1])
    
  if data_feature is not None:
    datapoints = data_feature
    if show_ref_images:
      assert img_path is not None
      filename = img_path

  else:  
    datapoints = torch.load(path[0])
    filename = path[1]

  points = torch.full((datapoints.shape[0], datapoints.shape[1], datapoints.shape[2]-1), 0.)
  data_full = datapoints[0]

  if torch.all(data_full == 0):
    return points

  if show_ref_images:
    if(filename[-3:] not in ['jpg', 'png']):
      return
    img = cv2.imread(f'{filename}')
    img_copy = img.copy()
    cv2_imshow(img_copy)

    # Draw human poses on image
    for face_points in [data_full[POIS[0]], data_full[POIS[1]]]:
        x = int(face_points[0])
        y = int(face_points[1])
        if x>0 and y>0:
          cv2.circle(img_copy, (x,y), radius=5, color=(0, 0, 255), thickness=-3)

    # Visualize Image
    cv2_imshow(img_copy)

  lx, ly, _ = data_full[POIS[0]]    
  rx, ry, _ = data_full[POIS[1]]   
  dx = rx-lx
  dy = ry-ly

  if dx != 0:
    tan = dy/dx
    theta = np.arctan(tan.cpu())
  else:
    theta = 0. 
  
  dist =  torch.sqrt(((data_full[POIS[0]][:2] - data_full[POIS[1]][:2])**2).sum(axis=0).float())
  if dist == 0:
    return points
  ratio = float(ideal_dist/dist)
  
  width = int(w * ratio)
  height = int(h * ratio)
  
  (cX, cY) = (width // 2, height // 2)
  M = cv2.getRotationMatrix2D((cX, cY), np.degrees(theta), 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1]) 
  nW = int((height * sin) + (width * cos))
  nH = int((height * cos) + (width * sin)) 
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY 

  # rotation
  theta = -theta
  A = np.matrix([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])

  data = np.float32(data_full[:,:2].cpu())

  # strip of 0s and -100s
  ind = np.logical_and(np.array(data).sum(axis=1) != 0, np.array(data).sum(axis=1) != -200.)

  data = data[np.array(data).sum(axis=1) != 0.]
  data = data[np.array(data).sum(axis=1) != -200.]

  data *= (ratio)
  data_c = data-np.array([cX, cY])
  data_rotated = data_c @ A.T
  data_rotated += np.array([(nW/2, nH/2)])
  
  # crop  
  box = cv2.minAreaRect(np.int0(data_rotated))
  bbox = np.int0(cv2.boxPoints(box))
  data_cropped = data_rotated - [np.min(bbox[:,0]), np.min(bbox[:,1])]

  points[0][ind] = torch.Tensor(data_cropped)

  if show_ref_images:
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    img_rot = cv2.warpAffine(img, M, (nW, nH))

    img_copy = img_rot.copy()

    img_copy = cv2.drawContours(img_copy, [bbox], 0, (255, 0, 0), 2)
    # img_rot = cv2.circle(img_rot, (bbox[1][0], bbox[1][1]), 0, (255, 255, 0), 10)

    cv2_imshow(img_copy)

    img_crop = img_rot[np.min(bbox[:,1]):np.max(bbox[:,1]), np.min(bbox[:,0]):np.max(bbox[:,0]) ]    

    cv2_imshow(img_crop)
    img_copy = img_crop.copy()

    for face_points in data_cropped:
      face_points = np.array(face_points)
      x = int(face_points[0][0])
      y = int(face_points[0][1])
      if x>0 and y>0:
        cv2.circle(img_copy, (x,y), radius=3, color=(0, 0, 255), thickness=-3)

    # Visualize Image
    cv2_imshow(img_copy)

    return points, img_crop
  return points  

def align_crop_resize_per_seq(feat, data, img_size=None):
  # align crop and resize in a sequences of frames
  for i in range(data.shape[0]):
    face_frame = align_crop_resize_per_frame(feature=feat, data_feature=data[i], img_size=img_size)
    data[i] =F.pad(face_frame, (0,1), "constant", 0.)
  return data 

def flip_vec(feat_vec, image=None):
  # flips the POIs coordinates horizontally
  flip_vec = feat_vec 
  height, width = (360, 640)

  if image is not None:
    flip_image = cv2.flip(image, 1)
    height, width = image.shape[:2]

  col = width - flip_vec[0,:,0]
  flip_vec[0,:,0] =  col

  if image is not None: 
    return flip_vec, flip_image
  return flip_vec  

def rotate_vec(feat_vec, angle, image=None):
  feat_vec = feat_vec[0,:,:2]
  theta = math.radians(angle)
  if image is not None:
    height, width = image[:2]
  else:
    height, width = (360, 640)
  (cX, cY) = (width // 2, height // 2)
  M = cv2.getRotationMatrix2D((cX, cY), np.degrees(theta), 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1]) 
  nW = int((height * sin) + (width * cos))
  nH = int((height * cos) + (width * sin)) 
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  theta = -theta
  A = np.matrix([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
  
  data_c = feat_vec-np.array([cX, cY])

  data_rotated = data_c @ A.T
  data_rotated += np.array([(nW/2, nH/2)])

  if image is not None:
    img_rot = cv2.warpAffine(img, M, (nW, nH))
    return data_rotated, img_rot

  return data_rotated

def augment_dataset(feature_vec, augmentations):
  # dataset augmentations

  augs = augmentations
  aug_set = torch.Tensor(augs, feature_vec.shape[0], feature_vec.shape[1], feature_vec.shape[2], feature_vec.shape[3], feature_vec.shape[4])

  for s, seq in enumerate(feature_vec):
    for f, frame in enumerate(seq):
      # flip
      aug_set[0][s][f] = flip_vec(frame)
      # rotate -5
      if augmentations > 1:
        frame = frame.cpu()
        aug_set[1][s][f][0,:,:2] = rotate_vec(frame, -5)
        # rotate 5
      if augmentations > 2:  
        aug_set[2][s][f][0,:,:2] = rotate_vec(frame, 5)
      # rotate flip 3
      if augmentations > 3:
        aug_set[3][s][f][0,:,:2] = rotate_vec(aug_set[1][s][f], -3)
        # rotate flip -3
      if augmentations > 4:  
        aug_set[4][s][f][0,:,:2] = rotate_vec(aug_set[1][s][f], 3)

  return aug_set 

def plot_loss_iters(name, content_path, valid_freq, losses, losses_validation):
    # Plot and save the loss vs iterations graph
    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title=f'Loss Curve for {name}')
    ax.plot(range(len(losses[1:])), losses[1:], color='blue')
    ax.plot(range(0, len(losses_validation[1:]) * valid_freq, valid_freq), losses_validation[1:], color='orange')

    plt.show()
    fig.savefig(f"loss_vs_iterations - {name}.png")    

def record_performance(name, score, score_w):
  if not os.path.exists(f'{content_path}/models/model_performance.csv'):
    f = open(f'{content_path}/models/model_performance.csv', 'w')     
    row = ('model_name', 'performance on test set - score', 'performance on test set weighted score')
  else:
    f = open(f'{content_path}/models/model_performance.csv', '+a')
    row = (name, score)
   
  writer = csv.writer(f)
  writer.writerow(row)
  f.close()  
  
def gen_voice_graph_and_score(model_name, model_type = None, VA=False, show_graph=True):
  # old function for visualising performance of vocal models, 
  # can test the svm and random forrest models

  # audio processing
  audios = []
  path = f'{content_path}/dataset/aud_3s/test/'
  ratings = np.load(f"{content_path}/opinions/audio_ratings_labels_test.npy")
  ratings_dev = np.load(f"{content_path}/opinions/audio_ratings_devs_labels_test.npy")
  # ratings = pd.read_csv(f'{content_path}/dataset/aud_3s/test/test_labels_stddev.csv')
      
  print(ratings.shape)
  print(len(os.listdir(path)))

  score = 0
  score_w = 0

  idx=0
  for j, filename in enumerate(np.array(os.listdir(path))[:-1]):
    if(filename[-3:] == "wav"):

      if model_type is not None:
        if 'LSTM' in model_type or 'GRU' in model_type:
          voice_model = torch.load(f"{content_path}/models/{model_name}")
          clip = read(f'{content_path}/dataset/aud_3s/test/{filename}')
          features = librosa.feature.mfcc(y=clip[1].astype(float), sr=clip[0], hop_length=512, n_fft=2048).transpose().astype(np.float32)
          
          if VA:
            va, l = aT.file_regression(f'{path}/{filename}', f"{content_path}/models/va_voice_svm", "svm")
            p1d = (0, features[1] - len(va))
            va = F.pad(torch.Tensor(va).view(1,2), (0, 18), "constant", 0)
            features = torch.cat((va.to(device), torch.tensor(features).to(device)), dim=0)

          seq_dim, input_dim = features.shape
          
          audio = Variable(Tensor(features).view(-1, seq_dim, input_dim).to(device))    
          acc = voice_model(audio).squeeze(-1)
          acc = acc.item()

        elif model_type == 'svm':  
          v, l = aT.file_regression(f'{content_path}/dataset/aud_3s/test/{filename}', f"{content_path}/models/voice_svm", model_type)
          acc = v[0]

        elif model_type == 'randomforest':
          v, l = aT.file_regression(f'{content_path}/dataset/aud_3s/test/{filename}', f"{content_path}/models/voice_rdmforrest", model_type)
          acc = v[0]

        else:
          acc = 0
          print(model_name)   

        score_w += (ratings[j-2] - acc)**2 *(1/(1+ratings_dev[j-2]))
        score += (ratings[j-2] - acc)**2
        # score_w += (ratings[idx,1] - acc)**2 *(1/(1+ratings_dev[idx,2]))
        # score += (ratings[idx,1] - acc)**2
        idx+=1
        audios.append(acc)

  if show_graph:
    plt.figure(figsize=(12, 8))
    print('no of ratings', len(ratings))
    print('no of video clips', len(audios))
    plt.plot(range(len(audios)), audios, label = 'model')
    plt.plot(range(len(ratings)), ratings, label='true ratings')
    plt.plot(range(len(ratings_dev)), ratings_dev, label='dev')

    print(f'error for {model_name} on test set: {score.item()}')
    print(f'weighted error for {model_name} on test set: {score_w.item()}')
    plt.legend()
    plt.show()

  record_performance(model_name, score.item() ,score_w.item())
  return (score.item(), score_w.item()), audios  

def gen_face_graph_and_score(model_name, model_type = None, VA=True, frames_per_seq = 12, show_graph=True):
  return gen_feat_graph_and_score(model_name, "face", model_type, VA, frames_per_seq, show_graph)

def gen_pose_graph_and_score(model_name, model_type = None, VA=False, frames_per_seq = 12, show_graph=True):
  return gen_feat_graph_and_score(model_name, "pose", model_type, VA, frames_per_seq, show_graph)

def gen_feat_graph_and_score(model_name, feat, model_type = None, VA=True, frames_per_seq:int = 12, show_graph=True):

  test_dataset_feat = torch.load(f"{content_path}/dataset/{feat}_test_unfiltered")
  test_genders = torch.load(f"{content_path}/dataset/genders_test_slice")

  test_dataset_feat = test_dataset_feat.to(device)
  test_genders = test_genders.to(device)

  test_feat_ratings = torch.tensor(np.load(f"{content_path}/opinions/{feat}_ratings_labels_test_unfiltered.npy"))
  stds = torch.tensor(np.load(f"{content_path}/opinions/{feat}_ratings_devs_test_unfiltered.npy"))

  print("here", frames_per_seq)
  if VA:
    test_va = torch.load(f"{content_path}/opinions/{feat}_va_best_test_slice_va")

    if  model_type != 'regression':
      test_va = torch.cat(test_va, 0)
      test_va = F.pad(test_va, (0, 1 ,0,(frames_per_seq- test_va.shape[0] % frames_per_seq)),  "constant", 0. )
      test_va = test_va.reshape(test_va.shape[0]//frames_per_seq, frames_per_seq, 1,1, 3)
      test_va=test_va.to(device)
  
  if model_type != 'regression':

    test_feat_ratings = F.pad(test_feat_ratings, (0,(frames_per_seq- test_feat_ratings.shape[0] % frames_per_seq)),  "constant", 0. )
    test_feat_ratings = test_feat_ratings.reshape(test_feat_ratings.shape[0]//frames_per_seq, frames_per_seq)
    # test_feat_ratings= torch.mean(test_feat_ratings, dim=1)
    test_feat_ratings.to(device)

    stds = F.pad(stds, (0,(frames_per_seq- stds.shape[0] % frames_per_seq)),  "constant", 0. )
    stds = stds.reshape(stds.shape[0]//frames_per_seq, frames_per_seq)
    # stds = torch.mean(stds, dim=1 )
    stds.to(device)

    test_dataset_feat = F.pad(test_dataset_feat, (0, 0, 0, 0, 0, 0, 0, (frames_per_seq- test_dataset_feat.shape[0] % frames_per_seq)),  "constant", 0. )
    test_dataset_feat = test_dataset_feat.reshape(test_dataset_feat.shape[0]//frames_per_seq, frames_per_seq, test_dataset_feat.shape[1], test_dataset_feat.shape[2], test_dataset_feat.shape[3])
    test_dataset_feat.to(device)

  model_feat = torch.load(f"{content_path}/models/{model_name}")
  model_feat = model_feat.to(device)

  out_feat = [] 

  score = 0
  score_w = 0
  count = 0

  mask =[]

  for f in range(test_dataset_feat.shape[0]):
    model_feat.eval()
    if not torch.all(test_dataset_feat[f] == 0) and torch.any(test_dataset_feat[f] > -99.0):
      with torch.no_grad():
        if model_type == 'regression':
          input = torch.cat((test_genders[f].float(), test_dataset_feat[f].to(device)), 1).flatten(start_dim=1)
          input = input.squeeze(-1)
          input = Variable(input.to(device))

        else: #LSTM or GRU
          if 'align' in model_name: 
            input = align_crop_resize_per_seq(feat, data=test_dataset_feat[f])
          #   face_frame= align_crop_resize_per_frame(data_feature=test_dataset_feat[f],feature=feat)
          #   test_dataset_feat[f] = F.pad(face_frame, (0,1), "constant", 0.)  
          if VA:
            input = torch.cat((test_va[f], input), 2)
          
          # print(input.shape)
          input = input.flatten(start_dim=1)
          seq_dim = input.shape[0]
          # input_dim = input.shape[-1]
          # seq_dim = test_dataset_feat[0].shape[0]
          input_dim = len(input[0].flatten())
          # print(input_dim)
          # print(seq_dim)
          input = Variable(input.view(-1, seq_dim, input_dim).to(device))

        out = model_feat(input.float())
        out_feat.append(out.cpu())
        score += (out - test_feat_ratings[f].cuda())**2
        score_w += ((out - test_feat_ratings[f].cuda())**2 * (1/(1 + stds[f].cuda())))

        count +=1
        mask.append(True)

    else:
      out_feat.append(-5)
      stds[f] = 0
      test_feat_ratings[f] = -5
      mask.append(False)

  if show_graph:
    plt.figure(figsize=(12, 8))
    stds = np.array(stds)
    out_feat = np.array(torch.cat(out_feat))
    test_feat_ratings = np.array(test_feat_ratings)
    
    # print('no of ratings', len(test_feat_ratings[mask]))
    # print('no of frames', len(out_feat[mask]))

    # plt.plot(range(len(out_feat[mask])), out_feat[mask], label = 'model predictions')
    # plt.plot(range(len(test_feat_ratings[mask])), test_feat_ratings[mask], label='true ratings')
    # plt.plot(range(len(stds[mask])), stds[mask], label='uncertainty level')

    print('no of ratings', len(test_feat_ratings[mask]))
    print('no of frames', len(out_feat[mask]))

    print(out_feat.flatten().shape)
    out_feat = (savgol_filter(np.array(out_feat.flatten()), 51, 3))
    # outs = out_feat.to(device)

    plt.plot(range(len(out_feat.flatten())), out_feat.flatten(), label = 'model predictions')
    plt.plot(range(len(test_feat_ratings.flatten())), test_feat_ratings.flatten(), label='true ratings')
    plt.plot(range(len(stds.flatten())), stds.flatten(), label='uncertainty level')

    print(f'error for {model_name} on test set: {score}')
    print(f'weighted error for {model_name} on test set: {score_w}')
    print('-----------------')
    plt.legend()
    plt.show()    

  record_performance(model_name, score, score_w)
  return (score, score_w), out_feat

def eval_single_cue(network_name, feat, test_dataset, smoothen, net_type="rnn", show_true_label=True):
  # currently used function for visualising performance of single cue models
  outs = []
  test_ratings =[]
  test_ratings_dev =[]

  score_MSE=0
  score_wMSE=0
  score_CCC=0
  
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)

  network = torch.load(f'{content_path}/models/{network_name}')
  network = network.to(device)
  network.eval()


  with torch.no_grad():
    # start_time = time.time()
    for iter, (feature, o, o_std) in enumerate(test_loader):

      if net_type != 'rnn':
        if feat == "voice":
          v, l = aT.file_regression(feature[0], f"{content_path}/models/voice_svm", "svm")
          network_prediction = torch.tensor(v[0])

        else:
          gender = torch.zeros((1,1,3)).cuda()
          if not torch.all (feature == 0) and torch.all (feature > -99.0):
            feature = feature.reshape(1,67,3).to(device)
            input = torch.cat((gender, feature.float()), 1).flatten(start_dim=1)
            network_prediction = network(input)
          else:
            network_prediction = torch.zeros((1,1)).cuda()  

      else:
        seq_dim, input_dim = feature.shape[1], feature.shape[2]
        input = Variable(feature.view(-1, seq_dim, input_dim).to(device))

      # else:  
      #   print(feature.shape)
      #   seq_dim = feature.shape[1]
      #   input_dim = len(feature[2].flatten())
        
      # input = Variable(feature.view(-1, seq_dim, input_dim).to(device))

        network_prediction = network(input)
      o = Variable(o.to(device))
      o_std = Variable(o_std.to(device))  

      outs.append(network_prediction.squeeze(0))
      test_ratings.append(o.squeeze(0))
      test_ratings_dev.append(o_std.squeeze(0))

  outs= torch.stack(outs)
  outs=outs.flatten()

  if smoothen:
    outs = torch.tensor(savgol_filter(np.array(outs.cpu()), 21, 3))
  outs = outs.to(device)

  # print(time.time() - start_time)

  test_ratings = torch.stack(test_ratings).flatten()
  test_ratings_dev = torch.stack(test_ratings_dev).flatten()

  plt.figure(figsize=(12, 8))


  loss = torch.nn.MSELoss(reduction='sum')(outs.float(), test_ratings.float())
  score_MSE+= loss.item() 

  # for i in range(0, len(outs)-batch_size, batch_size):
  #   loss = concordance_cc2(outs[i: i+batch_size], test_ratings[i: i+batch_size])
  #   score_CCC+=loss.item()
  # loss = concordance_cc2(outs[i+batch_size:], test_ratings[i+batch_size:])
  # score_CCC+=loss.item()   
  loss = concordance_cc2(outs, test_ratings)
  score_CCC+=loss.item() 

  scores_CCC_per_seq = []
  scores_wMSE_per_seq = []

  idxs = test_dataset.get_video_segmentation()

  ################
  # sequence concatenation
  ################

  print("here")
  print(idxs)
  if show_true_label and len(idxs) > 0:
    for i in range(len(idxs[:-1])):
      if feat == "voice":
        fr = idxs[i]
        next_fr = idxs[i+1]
      else:
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

  ##################    
      
  loss = torch.sum(2/(1+ test_ratings_dev) * (outs - test_ratings) ** 2)      
  score_wMSE+=loss.item()   
                                    
  print()
  # test_ratings = torch.stack(test_ratings).flatten()
  score = 1.0 - (torch.sum(torch.abs(outs - test_ratings) * (1/(1+test_ratings_dev)))) / len(outs)

  print(f"Weigthed score for {network_name} for {feat} after smoothening: {score}")
  print(f'CCC score for {network_name} for {feat} on test set: {score_CCC}')
  print()

  print(f'''CCC scores for {network_name} for {feat} on test set 
  on each video: {scores_CCC_per_seq}''')

  print(f'''w_MSE scores for {network_name} for {feat} on test set 
  on each video: {scores_wMSE_per_seq}''')


  plt.plot(range(outs.shape[0]), outs.cpu(), label = 'Model Predictions', marker='*')
  if show_true_label:
    plt.plot(range(len(test_ratings)), test_ratings.cpu(), label='True Ratings', marker = '.')

  # plt.plot(range(len(test_ratings_dev)), test_ratings_dev.cpu(), label='Uncertainty Factor')
  test_ratings_dev = test_ratings_dev.cpu()
  test_ratings = test_ratings.cpu()

  if show_true_label:
    plt.fill_between(range(len(test_ratings_dev)), test_ratings+test_ratings_dev/2, test_ratings-test_ratings_dev/2, facecolor='r', alpha=0.2)
  
  print()
  print(f'MSE error for {network_name} on test set: {score_MSE}')
  print(f'weighted MSE error for {network_name} on test set: {score_wMSE}')

  if show_true_label:
    plt.title(f"Single-cue Predictions of {feat.upper()} vs True {feat.upper()} Ratings")
  else:
    plt.title(f"Single-cue Predictions of {feat.upper()}")

  print(len(outs))
  print(len(test_ratings_dev))

  if show_true_label:
    try:
      for i, non in enumerate(torch.nonzero(test_dataset.get_non_available_frames().cpu())):
        plt.axvline(x=non, linewidth = 2, alpha = 0.2, color = 'black', 
        label='Features Not Detected' if i == 0 else "")
    except:
      print('no empty frames')    

  # seq_len of 1.5 sec for now
  repeat = len(outs)//10
  plt.xticks(range(0, int(len(outs)), repeat), [int(i * 1.5) for i in range(0, len(outs), repeat)])

  plt.xlabel('Time (seconds)')
  plt.ylabel('Overall Engagement')

  # score_overall = 1.0 - torch.sum(torch.abs(test_overall_ratings) * (1/(1 + test_overall_ratings_dev))) / len(outs)
  
  # print(f"overall score: {score_overall}")
  # print('-----------------')
  plt.legend()
  plt.show() 

  if net_type == 'regression':
    eval_fusion_score_group(outs.to(device), test_ratings.to(device), 
                            15 if feat == 'voice' else 15*9)

from bisect import bisect

groups = [0, -2.5, -1.25, 0, 1.5, 3, 5]

def eval_single_cue_with_score_classes(network_name, feat, test_dataset, smoothen, show_true_label=True):
  # create score group visualisation for single cure results
  outs = []
  test_ratings =[]
  test_ratings_dev =[]

  score_MSE=0
  score_wMSE=0
  score_CCC=0

  batch_size=1

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
  
  network = torch.load(f'{content_path}/models/{network_name}')
  network.eval()

  with torch.no_grad():
    for iter, (feature, o, o_std) in enumerate(test_loader):

      seq_dim, input_dim = feature.shape[1], feature.shape[2]
      input = Variable(feature.view(-1, seq_dim, input_dim).to(device))

      network_prediction = network(input)
      o = Variable(o.to(device))
      o_std = Variable(o_std.to(device))  

      outs.append(network_prediction.squeeze(0))
      test_ratings.append(o.squeeze(0))
      test_ratings_dev.append(o_std.squeeze(0))

  outs= torch.stack(outs)
  outs=outs.flatten()

  if smoothen:
    outs = torch.tensor(savgol_filter(np.array(outs.cpu()), 21, 3))
  outs = outs.to(device)

  test_ratings = torch.stack(test_ratings).flatten()
  test_ratings_dev = torch.stack(test_ratings_dev).flatten()

  # seq = 40
  seq = len(outs)//12
  outs_p = outs[: (len(outs)//seq) *seq]
  outs_short = outs[(len(outs)//seq) *seq:]

  outs = outs_p.reshape((len(outs_p)//seq, seq))
  outs = outs.mean(axis= 1)
  test_ratings_p = test_ratings[: (len(test_ratings)//seq) *seq]
  if outs_short.shape[0] > 0:

    outs = torch.cat((outs, outs_short.mean().unsqueeze(0)), 0)

  test_ratings_p = test_ratings[: (len(test_ratings)//seq) *seq]
  test_ratings_short = test_ratings[(len(test_ratings)//seq) *seq:]
  test_ratings = test_ratings_p.reshape((len(test_ratings_p)//seq, seq))
  test_ratings = test_ratings.mean(axis=1)
  if test_ratings_short.shape[0] > 0:
    test_ratings = torch.cat((test_ratings, test_ratings_short.mean().unsqueeze(0)), 0)

  outs = torch.tensor([groups[bisect(groups, i)] for i in outs]).to(device) 
  test_ratings = torch.tensor([groups[bisect(groups, i)] for i in test_ratings]).to(device)

  loss = torch.nn.MSELoss(reduction='sum')(outs.float(), test_ratings.float())
  score_MSE+= loss.item() 
                                    
  plt.figure(figsize=(12, 8))

  print()

  # plt.plot(range(outs.shape[0]), outs.cpu(), label = 'Model Predictions')
  # plt.plot(range(len(test_ratings)), test_ratings.cpu(), label='True Ratings')

  print(outs.shape)
  print(test_ratings.shape)

  # score = 1.0 - (torch.sum(torch.abs(outs - test_ratings) * (1/(1+test_ratings_dev)))) / len(outs)
  # print(f"Weigthed score for {network_name} for {feat} after smoothening: {score}")
  print(f'CCC score for {network_name} for {feat} on test set: {score_CCC}')
  print()

  for j, i in enumerate(groups):
    plt.axhline(y= i, color='g', alpha = 0.7, linestyle='-', label = 'score group'  if j == 0 else "")
  
  # plt.plot(range(len(test_ratings_dev)), test_ratings_dev.cpu(), label='Uncertainty Factor')
  test_ratings_dev = test_ratings_dev.cpu()
  test_ratings = test_ratings.cpu()

  # plt.fill_between(range(len(test_ratings_dev)), test_ratings+test_ratings_dev/2, test_ratings-test_ratings_dev/2, facecolor='r', alpha=0.2)
  unit = len(outs)-1
  for i, p in enumerate(outs):
    plt.axhspan(xmin=i/unit, xmax =(i+1)/unit, 
      ymin= groups[bisect(groups, p)-2], 
      ymax= groups[bisect(groups, p)-1],
      color='b', alpha = 0.5, hatch = '//',
      label='model prediction of score group' if i == 0 else "")
  
  if show_true_label:
    for i, p in enumerate(test_ratings):
      plt.axhspan(xmin=i/unit, xmax =(i+1)/unit, 
        ymin= groups[bisect(groups, p)-2], 
        ymax= groups[bisect(groups, p)-1],
        color='r', alpha = 0.35, hatch = '\\',
        label='true score group'  if i == 0 else "")
  
  print()
  print(f'MSE error for {network_name} on test set: {score_MSE}')
  print(f'weighted MSE error for {network_name} on test set: {score_wMSE}')

  plt.ylim(-3, 3.5)
  plt.title(f"Single-cue Predictions of {feat.upper()} vs True {feat.upper()} Ratings")
  print(len(outs))
  print(len(test_ratings_dev))

  # seq_len of 1.5 sec for now
  # plt.xticks(range(0, int(len(outs)), 10), [int(i * 1.5) for i in range(0, len(outs), 10)])
  plt.xticks(range(0, int(len(outs))), [int(i * seq *1.5) for i in range(0, len(outs))])

  plt.xlabel('Time (seconds)')
  plt.ylabel('Overall Engagement')

  plt.legend()
  plt.show() 

def eval_fusion_score_group(outs, test_ratings, seq=15, title=""):
  # create score group visualisation for set of results, called in fusion
  outs_p = outs[: (len(outs)//seq) *seq]
  outs_short = outs[(len(outs)//seq) *seq:]
  print(outs_short.shape)
  outs = outs_p.reshape((len(outs_p)//seq, seq))
  outs = outs.mean(axis= 1)

  if outs_short.shape[0] > 0:
    outs = torch.cat((outs, outs_short.mean().unsqueeze(0)), 0)

  test_ratings_p = test_ratings[: (len(test_ratings)//seq) *seq]
  test_ratings_short = test_ratings[(len(test_ratings)//seq) *seq:]
  test_ratings = test_ratings_p.reshape((len(test_ratings_p)//seq, seq))
  test_ratings = test_ratings.mean(axis=1)

  if test_ratings_short.shape[0] > 0:
    test_ratings = torch.cat((test_ratings, test_ratings_short.mean().unsqueeze(0)), 0)

  print([bisect(groups, i) for i in outs])
  print(outs)
  outs = torch.tensor([groups[bisect(groups, i)] for i in outs]).to(device) 
  test_ratings = torch.tensor([groups[bisect(groups, i)] for i in test_ratings]).to(device)

  loss = torch.nn.MSELoss(reduction='sum')(outs.float(), test_ratings.float())
  score_MSE = loss.item() 
                                    
  plt.figure(figsize=(12, 8))
  plt.title(title)
  print()

  # plt.plot(range(outs.shape[0]), outs.cpu(), label = 'Model Predictions')
  # plt.plot(range(len(test_ratings)), test_ratings.cpu(), label='True Ratings')

  print(outs.shape)
  print(test_ratings.shape)

  # score = 1.0 - (torch.sum(torch.abs(outs - test_ratings) * (1/(1+test_ratings_dev)))) / len(outs)
  # print(f"Weigthed score for {network_name} for {feat} after smoothening: {score}")
  # print(f'CCC score for {network_name} for {feat} on test set: {score_CCC}')
  print()

  for j, i in enumerate(groups):
    plt.axhline(y= i, color='g', alpha = 0.7, linestyle='-', label = 'score group'  if j == 0 else "")
  
  # plt.plot(range(len(test_ratings_dev)), test_ratings_dev.cpu(), label='Uncertainty Factor')
  # test_ratings_dev = test_ratings_dev.cpu()
  test_ratings = test_ratings.cpu()

  # plt.fill_between(range(len(test_ratings_dev)), test_ratings+test_ratings_dev/2, test_ratings-test_ratings_dev/2, facecolor='r', alpha=0.2)
  unit = len(outs)-1
  for i, p in enumerate(outs):
    plt.axhspan(xmin=i/unit, xmax =(i+1)/unit, 
      ymin= groups[bisect(groups, p)-2], 
      ymax= groups[bisect(groups, p)-1],
      color='b', alpha = 0.5, hatch = '//',
      label='model prediction of score group' if i == 0 else "")
  
  for i, p in enumerate(test_ratings):
    plt.axhspan(xmin=i/unit, xmax =(i+1)/unit, 
      ymin= groups[bisect(groups, p)-2], 
      ymax= groups[bisect(groups, p)-1],
      color='r', alpha = 0.35, hatch = '\\',
      label='true score group'  if i == 0 else "")
  
  plt.ylim(-3, 3.5)
  
  # seq_len of 1.5 sec for now
  # plt.xticks(range(0, int(len(outs)), 10), [int(i * 1.5) for i in range(0, len(outs), 10)])
  plt.xticks(range(0, int(len(outs))), [int(i * seq *1.5) for i in range(0, len(outs))])

  plt.xlabel('Time (seconds)')
  plt.ylabel('Overall Engagement')

  # score_overall = 1.0 - torch.sum(torch.abs(test_overall_ratings) * (1/(1 + test_overall_ratings_dev))) / len(outs)
  
  # print(f"overall score: {score_overall}")
  # print('-----------------')
  plt.legend()
  plt.show() 
