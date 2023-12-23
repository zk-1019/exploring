import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from rouge import Rouge
from bleu import *
from nltk.translate import meteor_score
from cider import *



import argparse
parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
parser.add_argument('--m', type=int, default=40)
parser.add_argument('--head', type=int, default=8)

# Parameters
data_folder = 'D:\\my-torch - sy11rsic\\caption data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'D:\\my-torch - sy11rsic\\BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = 'D:\\my-torch - sy11rsic\\caption data\\WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()


# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out,out1= encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)(1,14,14,2048)
        # atten_out = encoder_atteneion(encoder_out.view(-1,196,2048))
        enc_image_size = 14#(14)
        # encoder_dim = encoder_out1.size(3)#(2048)
        num_pixels = encoder_out.size(2)
        # Flatten encoding
        encoder_out = encoder_out.view(1, num_pixels, -1)  # (1, num_pixels, encoder_dim) (1,196,2048)
        # num_pixels = encoder_out.size(1) #(196)
        encoder_dim = encoder_out.size(-1)
        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out.view(k,4,196,512),out1, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            prev_word_inds = torch.tensor(prev_word_inds).type(torch.long)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)
    references_list = []
    hypotheses_list = []
    for i in references:
        wordss = []
        new = []
        for j in i:
            wordss = [rev_word_map[ind] for ind in j]
            new.append(wordss)
        references_list.append(new)
    wordss = []
    for i in hypotheses:
        wordss = [rev_word_map[ind] for ind in i]
        hypotheses_list.append(wordss)
        wordss = []
    # Calculate BLEU-4 scores
    j = 0
    metor_list = []
    for i in references_list:
        if j< len(hypotheses_list):
            # xxx = meteor_score.meteor_score([i[0],i[1],i[2],i[3],i[4]],hypotheses_list[j])
            xxx = round(meteor_score.meteor_score([i[0], i[1], i[2], i[3], i[4]], hypotheses_list[j]),4)
            j +=1
            metor_list.append(xxx)
    bleu4 = corpus_bleu(references, hypotheses)
    weights = (1.0/1.0, )
    bleu1 = corpus_bleu(references, hypotheses,weights)
    weights = (1.0 / 2.0, 1.0 / 2.0,)
    bleu2 = corpus_bleu(references, hypotheses, weights)
    weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
    bleu3 = corpus_bleu(references, hypotheses, weights)
    # weights = (1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0,)
    # # bleu5 = corpus_bleu(references, hypotheses, weights)
    scorer = Rouge()
    score, scores = scorer.compute_score(hypotheses, references)
    sum = 0
    for i in metor_list:
        sum+=i
    meteor = sum/(len(metor_list)*2)
    cider_scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score_cider, scores_cider) = cider_scorer.compute_score(hypotheses_list, references_list)
    return bleu1,bleu2,bleu3,bleu4,score,meteor,score_cider

if __name__ == '__main__':
    beam_size = 1
    b1,b2,b3,b4,rogue,meteor,cider = evaluate(beam_size)
    #print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
    print('\nbleu1 = {},\nb2 = {},\nb3 = {},\nb4 = {}'.format(b1,b2,b3,b4))
    print('rogue = {},\nmeteor = {},\ncider = {}'.format(rogue,meteor,cider))
    #print("bleu4.{},bleu4.{}".format(bb,bb1))