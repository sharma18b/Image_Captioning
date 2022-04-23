#!/usr/bin/env python3

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from transformer import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import codecs
import numpy as np


def train(args, train_loader, encoder, decodera, decoderb, decoderc, criterion, encoder_optimizer, decodera_optimizer, decoderb_optimizer, decoderc_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decodera.train()  # train mode (dropout and batchnorm is used)
    decoderb.train()  # train mode (dropout and batchnorm is used)
    decoderc.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    lossesa = AverageMeter()  # loss (per word decoded)
    lossesb = AverageMeter()  # loss (per word decoded)
    lossesc = AverageMeter()  # loss (per word decoded)
    top5accsa = AverageMeter()  # top5 accuracy
    top5accsb = AverageMeter()  # top5 accuracy
    top5accsc = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs,imgs1,imgs2 = imgs.to(device),imgs.to(device),imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs,imgs1,imgs2 = encoder(imgs)
        # print(imgs.shape)
        # print(imgs1.shape)
        # print(imgs2.shape)
        # dimension wise all imgs1, imgs , imgs2 have same dimensions 
        # imgs: [batch_size, 14, 14, 2048]
        # caps: [batch_size, 52]
        # caplens: [batch_size, 1]
        scoresa, caps_sorteda, decode_lengthsa, alphasa, sort_inda = decodera(imgs, caps, caplens)
        scoresb, caps_sortedb, decode_lengthsb, alphasb, sort_indb = decoderb(imgs1, caps, caplens)
        scoresc, caps_sortedc, decode_lengthsc, alphasc, sort_indc = decoderc(imgs2, caps, caplens)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targetsa = caps_sorteda[:, 1:]
        targetsb = caps_sortedb[:, 1:]
        targetsc = caps_sortedc[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scoresa = pack_padded_sequence(scoresa, decode_lengthsa, batch_first=True).data
        targetsa = pack_padded_sequence(targetsa, decode_lengthsa, batch_first=True).data
        scoresb = pack_padded_sequence(scoresb, decode_lengthsb, batch_first=True).data
        targetsb = pack_padded_sequence(targetsb, decode_lengthsb, batch_first=True).data
        scoresc = pack_padded_sequence(scoresc, decode_lengthsc, batch_first=True).data
        targetsc = pack_padded_sequence(targetsc, decode_lengthsc, batch_first=True).data
        # print(scores.size())
        # print(targets.size())

        # Calculate loss
        lossa = criterion(scoresa, targetsa)
        lossb = criterion(scoresb, targetsb)
        lossc = criterion(scoresc, targetsc)
        lossab = 0 #criterion(scoresa, scoresb)
        lossbc = 0 #criterion(scoresb, scoresc)
        lossca = 0 #criterion(scoresc, scoresa)
        # Add doubly stochastic attention regularization
        # Second loss, mentioned in paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
        # https://arxiv.org/abs/1502.03044
        # In section 4.2.1 Doubly stochastic attention regularization: We know the weights sum to 1 at a given timestep.
        # But we also encourage the weights at a single pixel p to sum to 1 across all timesteps T.
        # This means we want the model to attend to every pixel over the course of generating the entire sequence.
        # Therefore, we want to minimize the difference between 1 and the sum of a pixel's weights across all timesteps.
        if args.decoder_mode == "lstm":
            lossa += args.alpha_c * ((1. - alphasa.sum(dim=1)) ** 2).mean()
            lossb += args.alpha_c * ((1. - alphasb.sum(dim=1)) ** 2).mean()
            lossc += args.alpha_c * ((1. - alphasc.sum(dim=1)) ** 2).mean()
        elif args.decoder_mode == "transformer":
            print("transformer me ghus gya ")
            # dec_alphas = alphas["dec_enc_attns"]
            # alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
            # for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
            #     cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 52, 196]
            #     for h in range(args.n_heads):
            #         cur_head_alpha = cur_layer_alphas[:, h, :, :]
            #         loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

        # Back prop.
        decodera_optimizer.zero_grad()
        decoderb_optimizer.zero_grad()
        decoderc_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        lossa.backward(retain_graph = True)
        lossb.backward(retain_graph = True)
        lossc.backward(retain_graph = True)

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decodera_optimizer, args.grad_clip)
            clip_gradient(decoderb_optimizer, args.grad_clip)
            clip_gradient(decoderc_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        decodera_optimizer.step()
        decoderb_optimizer.step()
        decoderc_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5a = accuracy(scoresa, targetsa, 5)
        top5b = accuracy(scoresb, targetsb, 5)
        top5c = accuracy(scoresc, targetsc, 5)
        # top5ab = accuracy(scoresa, scoresb, 5)
        # top5bc = accuracy(scoresb, scoresc, 5)
        # top5ca = accuracy(scoresc, scoresa, 5)
        ## for a
        lossesa.update(lossa.item(), sum(decode_lengthsa))
        top5accsa.update(top5a, sum(decode_lengthsa))
        ## for a
        lossesb.update(lossb.item(), sum(decode_lengthsb))
        top5accsb.update(top5b, sum(decode_lengthsb))
        ## for a
        lossesc.update(lossc.item(), sum(decode_lengthsc))
        top5accsc.update(top5c, sum(decode_lengthsc))
        # ## for a
        # losses.update(loss.item(), sum(decode_lengths))
        # top5accs.update(top5, sum(decode_lengths))
        # ## for a
        # losses.update(loss.item(), sum(decode_lengths))
        # top5accs.update(top5, sum(decode_lengths))
        # ## for a
        # losses.update(loss.item(), sum(decode_lengths))
        # top5accs.update(top5, sum(decode_lengths))
        # ## for a
        # losses.update(loss.item(), sum(decode_lengths))
        # top5accs.update(top5, sum(decode_lengths))
        
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            print("Epoch: {}/{} step: {}/{} Loss-A: {} Loss-B: {} Loss-C: {} Loss-AB: {} Loss-BC: {} Loss-CA: {} AVG_Loss_A: {} AVG_Loss_B: {} AVG_Loss_C: {} Top-5 Accuracy_A: {} Top-5 Accuracy_B: {} Top-5 Accuracy_C: {} Batch_time: {}s".format(epoch+1, args.epochs, i+1, len(train_loader), lossesa.val, lossesb.val, lossesc.val, lossab, lossbc, lossca, lossesa.avg, lossesb.avg, lossesc.avg, top5accsa.val, top5accsb.val, top5accsc.val, batch_time.val))


def validate(args, val_loader, encoder, decodera,decoderb,decoderc, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """
    decodera.eval()  # eval mode (no dropout or batchnorm)
    decoderb.eval()  # eval mode (no dropout or batchnorm)
    decoderc.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    lossesa = AverageMeter()  # loss (per word decoded)
    lossesb = AverageMeter()  # loss (per word decoded)
    lossesc = AverageMeter()  # loss (per word decoded)
    top5accsa = AverageMeter()  # top5 accuracy
    top5accsb = AverageMeter()  # top5 accuracy
    top5accsc = AverageMeter()  # top5 accuracy

    start = time.time()

    referencesa = list()  # references (true captions) for calculating BLEU-4 score
    hypothesesa = list()  # hypotheses (predictions)
    referencesb = list()  # references (true captions) for calculating BLEU-4 score
    hypothesesb = list()  # hypotheses (predictions)
    referencesc = list()  # references (true captions) for calculating BLEU-4 score
    hypothesesc = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs,imgs1,imgs2 = encoder(imgs)
            scoresa, caps_sorteda, decode_lengthsa, alphasa, sort_inda = decodera(imgs, caps, caplens)
            scoresb, caps_sortedb, decode_lengthsb, alphasb, sort_indb = decoderb(imgs1, caps, caplens)
            scoresc, caps_sortedc, decode_lengthsc, alphasc, sort_indc = decoderc(imgs2, caps, caplens)
            # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            # targets = caps_sorted[:, 1:]
            targetsa = caps_sorteda[:, 1:]
            targetsb = caps_sortedb[:, 1:]
            targetsc = caps_sortedc[:, 1:]
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scoresa_copy = scoresa.clone()
            scoresb_copy = scoresb.clone()
            scoresc_copy = scoresc.clone()
            scoresa = pack_padded_sequence(scoresa, decode_lengthsa, batch_first=True).data
            targetsa = pack_padded_sequence(targetsa, decode_lengthsa, batch_first=True).data
            scoresb = pack_padded_sequence(scoresb, decode_lengthsb, batch_first=True).data
            targetsb = pack_padded_sequence(targetsb, decode_lengthsb, batch_first=True).data
            scoresc = pack_padded_sequence(scoresc, decode_lengthsc, batch_first=True).data
            targetsc = pack_padded_sequence(targetsc, decode_lengthsc, batch_first=True).data
            # scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            # targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            # Calculate loss
            lossa = criterion(scoresa, targetsa)
            lossb = criterion(scoresb, targetsb)
            lossc = criterion(scoresc, targetsc)
            lossab = criterion(scoresa, scoresb)
            lossbc = criterion(scoresb, scoresc)
            lossca = criterion(scoresc, scoresa)
            #  loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            if args.decoder_mode == "lstm":
                lossa += args.alpha_c * ((1. - alphasa.sum(dim=1)) ** 2).mean()
                lossb += args.alpha_c * ((1. - alphasb.sum(dim=1)) ** 2).mean()
                lossc += args.alpha_c * ((1. - alphasc.sum(dim=1)) ** 2).mean()
            elif args.decoder_mode == "transformer":
                print("transformer me ghus gya ")

            # Keep track of metrics
            # Keep track of metrics
            top5a = accuracy(scoresa, targetsa, 5)
            top5b = accuracy(scoresb, targetsb, 5)
            top5c = accuracy(scoresc, targetsc, 5)
            # top5ab = accuracy(scoresa, scoresb, 5)
            # top5bc = accuracy(scoresb, scoresc, 5)
            # top5ca = accuracy(scoresc, scoresa, 5)
            lossesa.update(lossa.item(), sum(decode_lengthsa))
            top5accsa.update(top5a, sum(decode_lengthsa))
            ## for a
            lossesb.update(lossb.item(), sum(decode_lengthsb))
            top5accsb.update(top5b, sum(decode_lengthsb))
            ## for a
            lossesc.update(lossc.item(), sum(decode_lengthsc))
            top5accsc.update(top5c, sum(decode_lengthsc))            
            # losses.update(loss.item(), sum(decode_lengths))
            # top5 = accuracy(scores, targets, 5)
            # top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcapsa = allcaps[sort_inda]  # because images were sorted in the decoder
            allcapsb = allcaps[sort_indb]  # because images were sorted in the decoder
            allcapsc = allcaps[sort_indc]  # because images were sorted in the decoder

            #  creating three hypothesis
            for j in range(allcapsa.shape[0]):
                img_caps = allcapsa[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                referencesa.append(img_captions)
            
            for j in range(allcapsb.shape[0]):
                img_caps = allcapsb[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                referencesb.append(img_captions)
            
            for j in range(allcapsc.shape[0]):
                img_caps = allcapsc[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                referencesc.append(img_captions)

            # Hypotheses A
            _, preds = torch.max(scoresa_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengthsa[j]])  # remove pads
            preds = temp_preds
            hypothesesa.extend(preds)
            # Hypotheses For B
            _, preds = torch.max(scoresb_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengthsb[j]])  # remove pads
            preds = temp_preds
            hypothesesb.extend(preds)
            # Hypotheses For C
            _, preds = torch.max(scoresc_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengthsc[j]])  # remove pads
            preds = temp_preds
            hypothesesc.extend(preds)

            assert len(referencesa) == len(hypothesesa)
            assert len(referencesb) == len(hypothesesb)
            assert len(referencesc) == len(hypothesesc)


    # Calculate BLEU-1~4 scores
    # metrics = {}
    # weights = (1.0 / 1.0,)
    # metrics["bleu1"] = corpus_bleu(references, hypotheses, weights)
    # weights = (1.0/2.0, 1.0/2.0,)
    # metrics["bleu2"] = corpus_bleu(references, hypotheses, weights)
    # weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
    # metrics["bleu3"] = corpus_bleu(references, hypotheses, weights)
    # metrics["bleu4"] = corpus_bleu(references, hypotheses)

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    metricsa = get_eval_score(referencesa, hypothesesa)
    metricsb = get_eval_score(referencesb, hypothesesb)
    metricsc = get_eval_score(referencesc, hypothesesc)

    print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} ROUGE_L {} CIDEr {}".format
          (lossesa.avg, top5accsa.avg,  metricsa["Bleu_1"],  metricsa["Bleu_2"],  metricsa["Bleu_3"],  metricsa["Bleu_4"],
           metricsa["ROUGE_L"], metricsa["CIDEr"]))
    print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} ROUGE_L {} CIDEr {}".format
          (lossesb.avg, top5accsb.avg,  metricsb["Bleu_1"],  metricsb["Bleu_2"],  metricsb["Bleu_3"],  metricsb["Bleu_4"],
           metricsb["ROUGE_L"], metricsb["CIDEr"]))
    print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} ROUGE_L {} CIDEr {}".format
          (lossesc.avg, top5accsc.avg,  metricsc["Bleu_1"],  metricsc["Bleu_2"],  metricsc["Bleu_3"],  metricsc["Bleu_4"],
           metricsc["ROUGE_L"], metricsc["CIDEr"]))

    return metricsa,metricsb,metricsc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')
    # Data parameters
    parser.add_argument('--data_folder', default="./dataset/generated_data",
                        help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="flickr30k_5_cap_per_img_5_min_word_freq",
                        help='base name shared by data files.')
    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of word embeddings.')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--decoder_mode', default="lstm", help='which model does decoder use?')  # lstm or transformer
    parser.add_argument('--attention_method', default="ByPixel", help='which attention method to use?')  # ByPixel or ByChannel
    parser.add_argument('--encoder_layers', type=int, default=2, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=6, help='the number of layers of decoder in Transformer.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=25, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=1, help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=False, help='whether fine-tune encoder or not')
    parser.add_argument('--fine_tune_embedding', type=bool, default=False, help='whether fine-tune word embeddings or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--embedding_path', default=None, help='path to pre-trained word Embedding.')
    args = parser.parse_args()

    # load checkpoint, these parameters can't be modified
    final_args = {"emb_dim": args.emb_dim,
                 "attention_dim": args.attention_dim,
                 "decoder_dim": args.decoder_dim,
                 "n_heads": args.n_heads,
                 "dropout": args.dropout,
                 "decoder_mode": args.decoder_mode,
                 "attention_method": args.attention_method,
                 "encoder_layers": args.encoder_layers,
                 "decoder_layers": args.decoder_layers}

    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    best_bleu4a = 0.  # BLEU-4 score right now
    best_bleu4b = 0.  # BLEU-4 score right now
    best_bleu4c = 0.  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print(device)

    # Read word map
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if args.checkpoint is None:
        encoder = CNN_Encoder(attention_method=args.attention_method)
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None

        if args.decoder_mode == "lstm":
            decodera = DecoderWithAttention(attention_dim=args.attention_dim,
                                           embed_dim=args.emb_dim,
                                           decoder_dim=args.decoder_dim,
                                           vocab_size=len(word_map),
                                           dropout=args.dropout)
            decoderb = DecoderWithAttention(attention_dim=args.attention_dim,
                                           embed_dim=args.emb_dim,
                                           decoder_dim=args.decoder_dim,
                                           vocab_size=len(word_map),
                                           dropout=args.dropout)
            decoderc = DecoderWithAttention(attention_dim=args.attention_dim,
                                           embed_dim=args.emb_dim,
                                           decoder_dim=args.decoder_dim,
                                           vocab_size=len(word_map),
                                           dropout=args.dropout)
        elif args.decoder_mode == "transformer":
            decoder = Transformer(vocab_size=len(word_map), embed_dim=args.emb_dim, encoder_layers=args.encoder_layers,
                                  decoder_layers=args.decoder_layers, dropout=args.dropout,
                                  attention_method=args.attention_method, n_heads=args.n_heads)

        decodera_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decodera.parameters()),
                                             lr=args.decoder_lr)
        decoderb_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoderb.parameters()),
                                             lr=args.decoder_lr)
        decoderc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoderc.parameters()),
                                             lr=args.decoder_lr)

        # load pre-trained word embedding
        if args.embedding_path is not None:
            all_word_embeds = {}
            for i, line in enumerate(codecs.open(args.embedding_path, 'r', 'utf-8')):
                s = line.strip().split()
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

            # change emb_dim
            args.emb_dim = list(all_word_embeds.values())[-1].size
            word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_map), args.emb_dim))
            for w in word_map:
                if w in all_word_embeds:
                    word_embeds[word_map[w]] = all_word_embeds[w]
                elif w.lower() in all_word_embeds:
                    word_embeds[word_map[w]] = all_word_embeds[w.lower()]
                else:
                    # <pad> <start> <end> <unk>
                    embedding_i = torch.ones(1, args.emb_dim)
                    torch.nn.init.xavier_uniform_(embedding_i)
                    word_embeds[word_map[w]] = embedding_i

            word_embeds = torch.FloatTensor(word_embeds).to(device)
            decodera.load_pretrained_embeddings(word_embeds)
            decodera.fine_tune_embeddings(args.fine_tune_embedding)
            decoderb.load_pretrained_embeddings(word_embeds)
            decoderb.fine_tune_embeddings(args.fine_tune_embedding)
            decoderc.load_pretrained_embeddings(word_embeds)
            decoderc.fine_tune_embeddings(args.fine_tune_embedding)
            print('Loaded {} pre-trained word embeddings.'.format(len(word_embeds)))

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['metrics']["Bleu_4"]
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decodera = checkpoint['decoder']
        decodera_optimizer = checkpoint['decoder_optimizer']
        decodera.fine_tune_embeddings(args.fine_tune_embedding)
        decoderb = checkpoint['decoder']
        decoderb_optimizer = checkpoint['decoder_optimizer']
        decoderb.fine_tune_embeddings(args.fine_tune_embedding)
        decoderc = checkpoint['decoder']
        decoderc_optimizer = checkpoint['decoder_optimizer']
        decoderc.fine_tune_embeddings(args.fine_tune_embedding)
        # load final_args from checkpoint
        final_args = checkpoint['final_args']
        for key in final_args.keys():
            args.__setattr__(key, final_args[key])
        if args.fine_tune_encoder is True and encoder_optimizer is None:
            print("Encoder_Optimizer is None, Creating new Encoder_Optimizer!")
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)

    # Move to GPU, if available
    decodera = decodera.to(device)
    decoderb = decoderb.to(device)
    decoderc = decoderc.to(device)
    encoder = encoder.to(device)
    # print("encoder_layers {} decoder_layers {} n_heads {} dropout {} attention_method {} encoder_lr {} "
    #       "decoder_lr {} alpha_c {}".format(args.encoder_layers, args.decoder_layers, args.n_heads, args.dropout,
    #                                         args.attention_method, args.encoder_lr, args.decoder_lr, args.alpha_c))
    # print(encoder)
    # print(decoder)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 25
        # 8 20
        if epochs_since_improvement == args.stop_criteria:
            print("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decodera_optimizer, 0.8)
            adjust_learning_rate(decoderb_optimizer, 0.8)
            adjust_learning_rate(decoderc_optimizer, 0.8)
            if args.fine_tune_encoder and encoder_optimizer is not None:
                print(encoder_optimizer)
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(args, train_loader=train_loader, encoder=encoder, decodera=decodera,decoderb=decoderb,decoderc=decoderc, 
            criterion=criterion, encoder_optimizer=encoder_optimizer, decodera_optimizer=decodera_optimizer, 
            decoderb_optimizer=decoderb_optimizer, decoderc_optimizer=decoderc_optimizer, epoch=epoch)

        # One epoch's validation
        metricsa, metricsb, metricsc = validate(args, val_loader=val_loader, encoder=encoder, decodera=decodera,decoderb=decoderb,decoderc=decoderc,
            criterion=criterion)
        recent_bleu4a = metricsa["Bleu_4"]
        recent_bleu4b = metricsb["Bleu_4"]
        recent_bleu4c = metricsc["Bleu_4"]

        # Check if there was an improvement
        is_besta = recent_bleu4a > best_bleu4a
        is_bestb = recent_bleu4b > best_bleu4b
        is_bestc = recent_bleu4c > best_bleu4c
        best_bleu4a = max(recent_bleu4a, best_bleu4a)
        if not is_besta:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, encoder, decodera, encoder_optimizer,
                        decodera_optimizer, metricsa, is_besta, final_args)
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, encoder, decoderb, encoder_optimizer,
                        decoderb_optimizer, metricsb, is_bestb, final_args)
        save_checkpoint(args.data_name, epoch, epochs_since_improvement, encoder, decoderc, encoder_optimizer,
                        decoderc_optimizer, metricsc, is_bestc, final_args)
