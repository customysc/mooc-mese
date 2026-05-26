import os
import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse
import pickle
from PIL import Image
from open_flamingo import create_model_and_transforms
import json
from collections import defaultdict
from utils import *
 
FLAMINGO_CONFIG = {
    'model_name': 'openflamingo/OpenFlamingo-9B-vitl-mpt7b',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'dim': 768  
}

def get_pretrained_emb(model, tokenizer, image_processor, sentences, entity_pos, eid2idx, 
                       entity_images, np_file, dim=768, batch_size=16):
 
    fp = np.memmap(np_file, dtype='float32', mode='w+', shape=(entity_pos[-1], dim))
    ptr_list = [0 for _ in entity_pos[:-1]]
    
    entity_sentences = defaultdict(list)
    for eid, sent in sentences:
        entity_sentences[eid].append(sent)
    
    for eid, sents in tqdm(entity_sentences.items(), desc="Processing entities"):
        if eid not in eid2idx:
            continue
            
        img_path = entity_images.get(eid, "")
        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
                vision_x = image_processor(image).unsqueeze(0).to(FLAMINGO_CONFIG['device'])
            except:
                vision_x = None
        else:
            vision_x = None
        
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:i+batch_size]
            
            lang_x = tokenizer(
                batch_sents, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).input_ids.to(FLAMINGO_CONFIG['device'])
            
            if vision_x is not None:
                vision_x_batch = vision_x.repeat(len(batch_sents), 1, 1, 1)
            else:
                vision_x_batch = None
            
            with torch.no_grad():
                outputs = model(
                    vision_x=vision_x_batch,
                    lang_x=lang_x,
                    attention_mask=(lang_x != tokenizer.pad_token_id).long(),
                    output_hidden_states=True
                )
                
                last_hidden_state = outputs.hidden_states[-1]
                
                cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
            
            for emb in cls_embeddings:
                this_idx = entity_pos[eid2idx[eid]] + ptr_list[eid2idx[eid]]
                ptr_list[eid2idx[eid]] += 1
                fp[this_idx] = emb.astype(np.float32)
    
    del fp
    return ptr_list

def load_entity_images(dataset_path, eid2name):
    image_mapping_path = os.path.join(dataset_path, "entity_images.pkl")
    if os.path.exists(image_mapping_path):
        return pickle.load(open(image_mapping_path, "rb"))
    
    entity_images = {}
    image_dir = os.path.join(dataset_path, "images")
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    for eid, name in eid2name.items():
        name_clean = name.replace(' ', '_').replace('/', '_').lower()
        possible_files = [
            f"{name_clean}.jpg",
            f"{name_clean}.png",
            f"{name_clean}.jpeg",
            f"{eid}.jpg",
            f"{eid}.png",
            f"{eid}.jpeg"
        ]
        
        found = False
        for file in possible_files:
            full_path = os.path.join(image_dir, file)
            if os.path.exists(full_path):
                entity_images[eid] = full_path
                found = True
                break
        
        if not found:
            default_path = os.path.join(image_dir, "default.jpg")
            if os.path.exists(default_path):
                entity_images[eid] = default_path
            else:
                blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
                blank_path = os.path.join(image_dir, "blank.jpg")
                blank_image.save(blank_path)
                entity_images[eid] = blank_path
    
    pickle.dump(entity_images, open(image_mapping_path, "wb"))
    return entity_images

def get_masked_sentences(sent_file, mask_token, eid2idx):
    sentences = []
    entity_count = {eid: 0 for eid in eid2idx}
    
    with open(sent_file, 'r') as f:
        data = json.load(f)
        
        for item in data:
            eid = item['entity_id']
            if eid not in eid2idx:
                continue
            masked_sent = item['sentence'].replace(item['mention'], mask_token)
            sentences.append((eid, masked_sent))
            entity_count[eid] += 1

    entity_pos = [0]
    for eid in sorted(eid2idx.keys()):
        if eid in entity_count:
            entity_pos.append(entity_pos[-1] + entity_count[eid])
    
    return sentences, entity_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-npy_out', default='pretrained_emb.npy', help='name of output npy file')
    parser.add_argument('-entity_pos_out', default='entity_pos.pkl', help='name of output entity index file')
    args = parser.parse_args()

    device = torch.device(FLAMINGO_CONFIG['device'])
    print(f"Using device: {device}")

    print("Loading OpenFlamingo model...")
    flamingo_model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="mosaicml/mpt-7b",
        tokenizer_path="mosaicml/mpt-7b",
        cross_attn_every_n_layers=4
    )
    flamingo_model.to(device)
    flamingo_model.eval()
    print("Model loaded successfully")

    eid2name, _, eid2idx = load_vocab(os.path.join(args.dataset, args.vocab))
    print(f"Loaded vocabulary with {len(eid2name)} entities")

    entity_images = load_entity_images(args.dataset, eid2name)
    print(f"Loaded images for {len(entity_images)} entities")

    sentences, entity_pos = get_masked_sentences(
        os.path.join(args.dataset, args.sent),
        tokenizer.mask_token,
        eid2idx
    )
    print(f"Loaded {len(sentences)} sentences")

    pickle.dump(entity_pos, open(os.path.join(args.dataset, args.entity_pos_out), 'wb'))
    print(f"Entity position data saved to {args.entity_pos_out}")

    print("Generating multimodal embeddings...")
    ptr_list = get_pretrained_emb(
        flamingo_model,
        tokenizer,
        image_processor,
        sentences,
        entity_pos,
        eid2idx,
        entity_images,
        np_file=os.path.join(args.dataset, args.npy_out),
        dim=FLAMINGO_CONFIG['dim']
    )
    
    print("\nEmbedding generation completed!")
    print(f"Embeddings saved to {args.npy_out}")
    print("Entities with embeddings:")
    for eid, idx in eid2idx.items():
        name = eid2name.get(eid, f"unknown_{eid}")
        count = ptr_list[idx]
        print(f"  {name} (ID: {eid}): {count} embeddings")
