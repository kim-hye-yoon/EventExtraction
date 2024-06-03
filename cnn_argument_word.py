import logging
import argparse
import sys
import time
import os
import pickle
import numpy as np

from utils import *
from models import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def make_prediction(inps, event_ids, start_triggers, end_triggers, sentence_ids, current_position_ids, id2event, id2argument):
    inps = inps.tolist()
    event_ids = [event_id[0] for event_id in event_ids.tolist()]
    start_triggers = start_triggers.tolist()
    end_triggers = end_triggers.tolist()
    sentence_ids = sentence_ids.tolist()
    current_position_ids = current_position_ids.tolist()

    argument_ids_of_sents = []
    argument_ids_of_sent = []
    event_id_of_sents = []
    start_triggers_of_sents =[]
    start_triggers_of_sent = []
    end_triggers_of_sents = []
    end_triggers_of_sent = []
    sentence_id_of_sents = []
    
    for i, current_position_id in enumerate(current_position_ids):
        if i == 0:
            argument_ids_of_sent = [inps[i]]
            start_triggers_of_sent = [start_triggers[i]]
            end_triggers_of_sent = [end_triggers[i]]
        else:
            if current_position_id == 0:
                argument_ids_of_sents.append(argument_ids_of_sent)
                event_id_of_sents.append(event_ids[i-1])
                start_triggers_of_sents.append(start_triggers_of_sent)
                end_triggers_of_sents.append(end_triggers_of_sent)
                sentence_id_of_sents.append(sentence_ids[i-1])

                argument_ids_of_sent = [inps[i]]
                start_triggers_of_sent = [start_triggers[i]]
                end_triggers_of_sent = [end_triggers[i]]
            else:
                argument_ids_of_sent.append(inps[i])
                start_triggers_of_sent.append(start_triggers[i])
                end_triggers_of_sent.append(end_triggers[i])

    argument_ids_of_sents.append(argument_ids_of_sent)
    event_id_of_sents.append(event_ids[len(event_ids)-1])
    start_triggers_of_sents.append(start_triggers_of_sent)
    start_trigger_of_sents = [start_triggers_of_sent[0] for start_triggers_of_sent in start_triggers_of_sents]
    end_triggers_of_sents.append(end_triggers_of_sent)
    end_trigger_of_sents = [end_triggers_of_sent[0] for end_triggers_of_sent in end_triggers_of_sents]
    sentence_id_of_sents.append(sentence_ids[len(sentence_ids)-1])
    
    result = []
    for i, argument_ids_of_sent in enumerate(argument_ids_of_sents):
        result_sent = []
        j = 0
        event_id = event_id_of_sents[i]
        event_type = id2event[event_id]
        start_trigger = start_trigger_of_sents[i]
        end_trigger = end_trigger_of_sents[i]
        sentence_id = sentence_id_of_sents[i]

        while j < len(argument_ids_of_sent):
            argument_id = argument_ids_of_sent[j]
            argument_type = id2argument[argument_id]
            if argument_type != 'O':
                tmp = [sentence_id, event_type, start_trigger, end_trigger, argument_type, j]
                while j+1 < len(argument_ids_of_sent) and argument_ids_of_sent[j+1] == argument_ids_of_sent[j]:
                    j += 1
                tmp.append(j)
                result_sent.append(tuple(tmp))
            j += 1
        result.extend(result_sent)
    
    return result

def convert_gold_arguments(arguments):
    result = []
    for sentence_id, argument_sent in enumerate(arguments):
        for argument in argument_sent:
            event_type = argument[0]
            start_trigger = argument[1]
            end_trigger = argument[2]
            for argument_tuple in argument[3:]:
                argument_type = argument_tuple[0]
                start_argument = argument_tuple[1]
                end_argument = argument_tuple[2]
                tmp = (sentence_id, event_type, start_trigger, end_trigger, argument_type, start_argument, end_argument)
                result.append(tmp)

    return result

def evaluate_for_train(preds, labels):
    
    preds = preds.tolist()
    labels = labels.tolist()

    num_labels, num_preds, num_true_positives = 0.0, 0.0, 0.0
    for i in range(len(preds)):
        if preds[i] == 0 and labels[i] == 0:
            continue
        elif preds[i] != 0 and labels[i] == 0:
            num_preds += 1
        elif preds[i] == 0 and labels[i] != 0:
            num_labels += 1
        else:
            num_preds += 1
            num_labels += 1
            if preds[i] == labels[i]:
                num_true_positives += 1
    
    f1, precision, recall = 0, 0, 0
    if num_preds != 0:
        precision = 100.0 * num_true_positives / num_preds
    else:
        precision = 0
    if num_labels != 0:
        recall = 100.0 * num_true_positives / num_labels
    else:
        recall = 0
    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall

def evaluate_for_test(preds, labels):

    num_preds = len(preds)
    num_labels = len(labels)
    num_true_positives = 0.0

    for pred in preds:
        for label in labels:
            if pred == label:
                num_true_positives += 1
                break

    f1, precision, recall = 0, 0, 0
    if num_preds != 0:
        precision = 100.0 * num_true_positives / num_preds
    else:
        precision = 0
    if num_labels != 0:
        recall = 100.0 * num_true_positives / num_labels
    else:
        recall = 0
    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    log_folder = 'log/argument_detection'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    weight_folder = 'weight/argument_detection'
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    
    out_folder = 'out/argument_detection'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    log_file = f'cnn-argument-word-{args.use_pretrained}-word2vec-{args.use_postag}-postag.log'
    logger.addHandler(logging.FileHandler(os.path.join(log_folder, log_file)))

    logger.info(str(args))

    train_sents, train_postags_of_sents, train_events_of_sents, train_arguments_of_sents, train_max_length \
        = convert_dataset_to_list('data/train')
    dev_sents, dev_postags_of_sents, dev_events_of_sents, dev_arguments_of_sents, dev_max_length \
        = convert_dataset_to_list('data/dev')
    test_sents, test_postags_of_sents, test_events_of_sents, test_arguments_of_sents, test_max_length \
        = convert_dataset_to_list('data/test')
    max_length = max(train_max_length, dev_max_length, test_max_length)
    
    word2id, id2word = build_vocab_word(train_sents)
    postag2id, id2postag = build_vocab_postag(train_postags_of_sents)
    event2id, id2event = build_vocab_event(train_events_of_sents)
    argument2id, id2argument = build_vocab_argument(train_arguments_of_sents)


    train_data = ArgumentWordDataset(train_sents, train_arguments_of_sents, train_postags_of_sents, \
        word2id, argument2id, event2id, postag2id, max_length)
    dev_data = ArgumentWordDataset(dev_sents, dev_arguments_of_sents, dev_postags_of_sents, \
        word2id, argument2id, event2id, postag2id, max_length)
    test_data = ArgumentWordDataset(test_sents, test_arguments_of_sents, test_postags_of_sents, \
        word2id, argument2id, event2id, postag2id, max_length)

    train_data_loader = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True)
    dev_data_loader = DataLoader(dataset=dev_data, batch_size=args.eval_batch_size, shuffle=False)
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.eval_batch_size, shuffle=False)

    lr = args.lr

    embedding_weight = None
    if args.use_pretrained:
        embedding_weight = np.load('weight/word2vec.npy')
        embedding_weight = torch.FloatTensor(embedding_weight)

    if not args.use_postag:
        model = ArgumentWordCNN(
            num_labels=len(argument2id),
            num_words=len(word2id),
            word_embedding_dim=args.word_embedding_dim,
            max_length=max_length,
            position_embedding_dim=args.position_embedding_dim,
            num_events=len(event2id),
            event_embedding_dim=args.event_embedding_dim,
            kernel_sizes=args.kernel_sizes,
            num_filters=args.num_filters,
            dropout_rate=args.dropout_rate,
            use_pretrained=args.use_pretrained,
            embedding_weight=embedding_weight
        )
    else:
        model = ArgumentWordCNNWithPostag(
            num_labels=len(argument2id),
            num_words=len(word2id),
            word_embedding_dim=args.word_embedding_dim,
            max_length=max_length,
            position_embedding_dim=args.position_embedding_dim,
            num_events=len(event2id),
            event_embedding_dim=args.event_embedding_dim,
            num_postags=len(postag2id),
            postag_embedding_dim=args.postag_embedding_dim,
            kernel_sizes=args.kernel_sizes,
            num_filters=args.num_filters,
            dropout_rate=args.dropout_rate,
            use_pretrained=args.use_pretrained,
            embedding_weight=embedding_weight
        )
    model.to(device)

    if not args.no_train:
        optimizer = Adam(model.parameters(), lr=lr)

        max_dev_f1 = 0
        for epoch in range(args.num_epochs):
            train_loss = 0
            train_f1 = 0
            train_precision = 0
            train_recall = 0
            nb_train_steps = 0

            start_time = time.time()

            model.train()
            logger.info(f'Epoch {epoch+1}|{args.num_epochs}:')
            for (batch, data) in enumerate(train_data_loader):
                input_ids = data['input_ids'].to(device)
                candidate_position_ids = data['candidate_position_ids'].to(device)
                trigger_position_ids = data['trigger_position_ids'].to(device)
                postag_ids = data['postag_ids'].to(device)
                event_ids = data['event_ids'].to(device)
                start_triggers = data['start_trigger'].to(device)
                end_triggers = data['end_trigger'].to(device)
                
                labels = data['argument_id'].to(device)
                sentence_ids = data['sentence_id'].to(device)
                current_position_ids = data['current_position_id'].to(device)
                
                if not args.use_postag:
                    loss, logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, labels)
                else:
                    loss, logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, postag_ids, labels)
                
                train_loss += loss.item()
                nb_train_steps += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                preds = torch.argmax(logits, dim=-1)
                f1, precision, recall = evaluate_for_train(preds, labels)
                
                train_f1 += f1
                train_precision += precision
                train_recall += recall
            
                if batch % args.log_step == 0 or batch+1 == len(train_data_loader):
                    logger.info(f'Batch {batch+1}|{len(train_data_loader)}: loss {loss.item():.4f} f1 {f1:.2f} precision {precision:.2f} recall {recall:.2f}')
        
            logger.info(f'Train: loss {train_loss/nb_train_steps:.4f} f1 {train_f1/nb_train_steps:.2f} precision {train_precision/nb_train_steps:.2f} recall {train_recall/nb_train_steps:.2f}')
            logger.info(f'Time: {time.time() - start_time:.2f}')


            dev_loss = 0
            nb_dev_steps = 0
            list_logits = []
            list_event_ids = [] 
            list_start_triggers = []
            list_end_triggers = []
            list_sentence_ids = []
            list_current_position_ids = []

            model.eval()
            for (batch, data) in enumerate(dev_data_loader):
                input_ids = data['input_ids'].to(device)
                candidate_position_ids = data['candidate_position_ids'].to(device)
                trigger_position_ids = data['trigger_position_ids'].to(device)
                postag_ids = data['postag_ids'].to(device)
                event_ids = data['event_ids'].to(device)
                start_triggers = data['start_trigger'].to(device)
                end_triggers = data['end_trigger'].to(device)
                
                labels = data['argument_id'].to(device)
                sentence_ids = data['sentence_id'].to(device)
                current_position_ids = data['current_position_id'].to(device)
                
                with torch.no_grad():
                    if not args.use_postag:
                        loss, logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, labels)
                    else:
                        loss, logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, postag_ids, labels)
                    
                dev_loss += loss.item()
                nb_dev_steps += 1

                list_logits.append(logits)
                list_event_ids.append(event_ids)
                list_start_triggers.append(start_triggers)
                list_end_triggers.append(end_triggers)
                list_sentence_ids.append(sentence_ids)
                list_current_position_ids.append(current_position_ids)
            
            logits = torch.cat(list_logits, dim=0)
            event_ids = torch.cat(list_event_ids, dim=0)
            start_triggers = torch.cat(list_start_triggers, dim=0)
            end_triggers = torch.cat(list_end_triggers, dim=0)
            sentence_ids = torch.cat(list_sentence_ids, dim=0)
            current_position_ids = torch.cat(list_current_position_ids, dim=0)

            preds = torch.argmax(logits, dim=-1)
            preds = make_prediction(preds, event_ids, start_triggers, end_triggers, \
                sentence_ids, current_position_ids, id2event, id2argument)
            labels = convert_gold_arguments(dev_arguments_of_sents)
            dev_f1, dev_precision, dev_recall = evaluate_for_test(preds, labels)

            logger.info(f'Dev: loss {dev_loss/nb_dev_steps:.4f} f1 {dev_f1:.2f} precision {dev_precision:.2f} recall {dev_recall:.2f}')

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                torch.save(model.state_dict(), os.path.join(weight_folder, f'cnn-argument-word-{args.use_pretrained}-word2vec-{args.use_postag}-postag.pth'))
                logger.info(f'Save model weight!')
        
            logger.info('')
    
    # result on test
    model.load_state_dict(torch.load(os.path.join(weight_folder, f'cnn-argument-word-{args.use_pretrained}-word2vec-{args.use_postag}-postag.pth'), map_location=device))
    logger.info('Restore best model !')
    
    list_logits = []
    list_event_ids = [] 
    list_start_triggers = []
    list_end_triggers = []
    list_sentence_ids = []
    list_current_position_ids = []

    model.eval()
    for (batch, data) in enumerate(test_data_loader):
        input_ids = data['input_ids'].to(device)
        candidate_position_ids = data['candidate_position_ids'].to(device)
        trigger_position_ids = data['trigger_position_ids'].to(device)
        postag_ids = data['postag_ids'].to(device)
        event_ids = data['event_ids'].to(device)
        start_triggers = data['start_trigger'].to(device)
        end_triggers = data['end_trigger'].to(device)
                
        labels = data['argument_id'].to(device)
        sentence_ids = data['sentence_id'].to(device)
        current_position_ids = data['current_position_id'].to(device)

        with torch.no_grad():        
            if not args.use_postag:
                logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids)
            else:
                logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, postag_ids)

        list_logits.append(logits)
        list_event_ids.append(event_ids)
        list_start_triggers.append(start_triggers)
        list_end_triggers.append(end_triggers)
        list_sentence_ids.append(sentence_ids)
        list_current_position_ids.append(current_position_ids)
            
    logits = torch.cat(list_logits, dim=0)
    event_ids = torch.cat(list_event_ids, dim=0)
    start_triggers = torch.cat(list_start_triggers, dim=0)
    end_triggers = torch.cat(list_end_triggers, dim=0)
    sentence_ids = torch.cat(list_sentence_ids, dim=0)
    current_position_ids = torch.cat(list_current_position_ids, dim=0)

    preds = torch.argmax(logits, dim=-1)
    preds = make_prediction(preds, event_ids, start_triggers, end_triggers, \
        sentence_ids, current_position_ids, id2event, id2argument)
    labels = convert_gold_arguments(test_arguments_of_sents)
    test_f1, test_precision, test_recall = evaluate_for_test(preds, labels)
    
    logger.info(f'Test: f1 {test_f1:.2f} precision {test_precision:.2f} recall {test_recall:.2f}')
    
    if args.eval_predicted_trigger:
        with open(args.predicted_trigger_file, 'rb') as f:
            predicted_events_of_sents = pickle.load(f)
        
        predicted_test_data = ArgumentWordDataset(test_sents, predicted_events_of_sents, test_postags_of_sents, \
            word2id, argument2id, event2id, postag2id, max_length)
        predicted_test_data_loader = DataLoader(dataset=predicted_test_data, batch_size=args.eval_batch_size, shuffle=False)

        list_logits = []
        list_event_ids = [] 
        list_start_triggers = []
        list_end_triggers = []
        list_sentence_ids = []
        list_current_position_ids = []

        model.eval()
        for batch, data in enumerate(predicted_test_data_loader):
            input_ids = data['input_ids'].to(device)
            candidate_position_ids = data['candidate_position_ids'].to(device)
            trigger_position_ids = data['trigger_position_ids'].to(device)
            postag_ids = data['postag_ids'].to(device)
            event_ids = data['event_ids'].to(device)
            start_triggers = data['start_trigger'].to(device)
            end_triggers = data['end_trigger'].to(device)
                
            labels = data['argument_id'].to(device)
            sentence_ids = data['sentence_id'].to(device)
            current_position_ids = data['current_position_id'].to(device)

            with torch.no_grad():        
                if not args.use_postag:
                    logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids)
                else:
                    logits = model(input_ids, candidate_position_ids, trigger_position_ids, event_ids, postag_ids)
                
            list_logits.append(logits)
            list_event_ids.append(event_ids)
            list_start_triggers.append(start_triggers)
            list_end_triggers.append(end_triggers)
            list_sentence_ids.append(sentence_ids)
            list_current_position_ids.append(current_position_ids)

        logits = torch.cat(list_logits, dim=0)
        event_ids = torch.cat(list_event_ids, dim=0)
        start_triggers = torch.cat(list_start_triggers, dim=0)
        end_triggers = torch.cat(list_end_triggers, dim=0)
        sentence_ids = torch.cat(list_sentence_ids, dim=0)
        current_position_ids = torch.cat(list_current_position_ids, dim=0)

        preds = torch.argmax(logits, dim=-1)
        preds = make_prediction(preds, event_ids, start_triggers, end_triggers, \
            sentence_ids, current_position_ids, id2event, id2argument)
        labels = convert_gold_arguments(test_arguments_of_sents)
        
        test_f1, test_precision, test_recall = evaluate_for_test(preds, labels)

        logger.info(f'Predicted trigger:\nTest: f1 {test_f1:.2f} precision {test_precision:.2f} recall {test_recall:.2f}\n')

        with open(os.path.join(out_folder, f'cnn-argument-word-{args.use_pretrained}-word2vec-{args.use_postag}-postag.pkl'), 'wb') as f:
            pickle.dump(preds, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--eval_predicted_trigger', action='store_true')
    parser.add_argument('--predicted_trigger_file', default=None, type=str)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--log_step', default=500, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--word_embedding_dim', default=300, type=int)
    parser.add_argument('--position_embedding_dim', default=25, type=int)
    parser.add_argument('--event_embedding_dim', default=25, type=int)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--use_postag', action='store_true')
    parser.add_argument('--postag_embedding_dim', default=25, type=int)
    parser.add_argument('--kernel_sizes', default=[2,3,4,5], type=list)
    parser.add_argument('--num_filters', default=150, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)

    args = parser.parse_args()

    main(args)