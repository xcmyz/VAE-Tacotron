import torch
import torch.nn as nn
from torch import optim

from network import *
from gen_training_loader import DataLoader, collate_fn, SpeechData
from loss import VAE_TacotronLoss
# from vae import ReferenceEncoder
# from vae import VAE

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = Tacotron().to(device)
    print("Model Have Been Defined")
    
    # Print info
    print("Batch Size: %d" % args.batch_size)
    print("Epoch Size: %d" % hp.epochs)
    # print()
    # print(model)

    # Get dataset
    dataset = SpeechData(args.dataset_path)
    # print(type(args.dataset_path))
    # print(len(dataset))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # # Loss for frequency of human register
    # n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    # Criterion
    criterion = VAE_TacotronLoss(device)

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True, num_workers=cpu_count())
    # print(len(training_loader))

    # Load logger
    f_logger = open("logger.txt", "w")
    f_logger.close()

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("---Model Restored at Step %d---" % args.restore_step)

    except:
        print("---Start New Training---")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Training
    model = model.train()

    total_step = hp.epochs * len(training_loader)
    # print(total_step)
    # Loss = []
    Time = np.array([])
    Start = time.perf_counter()
    for epoch in range(hp.epochs):
        # print("########")
        for i,  data_batch in enumerate(training_loader):
            start_time = time.perf_counter()
            # print("in")
            # Count step
            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1
            # print(current_step)

            # Init
            optimizer.zero_grad()

            #  {"text": texts, "mel": mels, "spec": specs}
            texts = data_batch["text"]
            # print(texts)
            mels = trans(data_batch["mel"])
            # mels = trans(mels)
            # print(np.shape(mels))
            specs = trans(data_batch["spec"])
            # print(np.shape(specs))
            # mel_input = mels[:, :-1, :]
            # print(np.shape(mel_input))
            # mel_input = mel_input[:, :, -hp.num_mels:]
            # print(np.shape(mel_input))
            # print(np.shape(mels))
            frame_arr = np.zeros(
                [args.batch_size, hp.num_mels, 1], dtype=np.float32)
            # print(np.shape(frame_arr))
            # print(np.shape(mels[:, :, 1:]))
            mel_input = np.concatenate((frame_arr, mels[:, :, 1:]), axis=2)
            # print(np.shape(mel_input))
            # print(mels)

            if torch.cuda.is_available():
                texts = torch.from_numpy(texts).type(
                    torch.cuda.LongTensor).to(device)
            else:
                texts = torch.from_numpy(texts).type(
                    torch.LongTensor).to(device)
            mels = torch.from_numpy(mels).to(device)
            specs = torch.from_numpy(specs).to(device)
            mel_input = torch.from_numpy(mel_input).to(device)

            # Forward

            ##############################
            # mel_input: (batch, 80, seq_length)
            ##############################
            # print(np.shape(mel_input))

            # # Test ReferenceEncoder
            # # ReferenceEncoder_Test = ReferenceEncoder().to(device)
            # vae = VAE().to(device)
            # # print(np.shape(torch.transpose(mel_input, 1, 2)))
            # # print(np.shape(ReferenceEncoder_Test(torch.transpose(mel_input, 1, 2))))
            # print(np.shape(vae(mel_input)))

            mel_output, linear_output, mu, log_var = model.forward(
                texts, mel_input)
            # print("#####################")
            # print(np.shape(mel_output))
            # print(np.shape(linear_output))
            # print()
            # print(np.shape(mels[:, :, 1:]))
            # print(np.shape(np.transpose(mels.cpu().numpy())))
            # print(np.shape(specs))
            # print(np.shape(np.transpose(mels)))

            # Calculate loss
            # st = time.clock()
            ##############################
            # mels: (batch, 80, seq_length)
            ##############################

            mel_loss, linear_loss, kl_div = criterion(
                mel_output, mels[:, :, 1:], linear_output, specs, mu, log_var)

            # mel_loss = torch.abs(
            #     mel_output - compare(mel_output, mels[:, :, 1:], device))
            # mel_loss = torch.mean(mel_loss)
            # linear_loss = torch.abs(
            #     linear_output - compare(linear_output, specs, device))
            # linear_loss = torch.mean(linear_loss)
            loss = mel_loss + 0.5 * linear_loss + kl_div
            # print(loss)
            # loss = loss.to(device)
            # Loss.append(loss)
            # et = time.clock()
            # print(et - st)
            # print(loss)

            # Backward
            loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            if current_step % hp.log_step == 0:
                Now = time.perf_counter()
                # print("time per step: %.2f sec" % time_per_step)
                # print("At timestep %d" % current_step)
                # print("linear loss: %.4f" % linear_loss.data[0])
                # print("mel loss: %.4f" % mel_loss.data[0])
                # print("total loss: %.4f" % loss.data[0])
                str_loss = "Epoch [{}/{}], Step [{}/{}], Linear Loss: {:.4f}, Mel Loss: {:.4f}, KL Loss: {:.6f}, Total Loss: {:.4f}.".format(
                    epoch+1, hp.epochs, current_step, total_step, linear_loss.item(), mel_loss.item(), kl_div.item(), loss.item())
                str_time = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))
                print(str_loss)
                print(str_time)
                with open("logger.txt", "a") as f_logger:
                    f_logger.write(str_loss + "\n")
                    f_logger.write(str_time + "\n")

            # print(current_step)
            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)
                # print(Time)


def trans(arr):
    return np.stack([np.transpose(ele) for ele in arr])
    # for i, b in enumerate(arr):
    # arr[i] = np.transpose(b)


def adjust_learning_rate(optimizer, step):
    if step == 500000:
        # if step == 20:
        # print("update")
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


# def compare(out, stan, device):
#     # for batch_index in range(len(out)):
#     #     for i in range(min([np.shape(out)[2], np.shape(stan)[2]])):
#     #         torch.abs(out[batch_index][i], stan[batch_index][i])
#     # cnt = min([np.shape(out)[2], np.shape(stan)[2]])
#     if np.shape(stan)[2] >= np.shape(out)[2]:
#         return stan[:, :, :np.shape(out)[2]]
#     # return out[:,:,:cnt], stan[:,:,:cnt]
#     else:
#         frame_arr = np.zeros([np.shape(out)[0], np.shape(out)[1], np.shape(out)[
#                              2]-np.shape(stan)[2]], dtype=np.float32)
#         return torch.Tensor(np.concatenate((stan.cpu(), frame_arr), axis=2)).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='dataset path', default='dataset')
    # parser.add_argument('--restore_step', type=int,
    #                     help='Global step to restore checkpoint', default=0)
    # parser.add_argument('--batch_size', type=int,
    #                     help='Batch size', default=hp.batch_size)

    # Test
    parser.add_argument('--batch_size', type=int, help='Batch size', default=2)
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=0)

    args = parser.parse_args()
    main(args)
