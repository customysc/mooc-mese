import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import nltk
import inflect
from PIL import Image 
from open_flamingo import create_model_and_transforms
from collections import defaultdict as ddict
import random
from utils import apk

FLAMINGO_CONFIG = {
    'model_name': 'openflamingo/OpenFlamingo-9B-vitl-mpt7b',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'max_new_tokens': 10,   
    'temperature': 0.7,
    'num_beams': 3,
    'top_p': 0.9,
    'threshold': 0.5 
}

GENERATION_SAMPLE_SIZE = 20
EXPANSION_SAMPLE_SIZE = 50
RANKING_TEMPLATES = 5

class CGExpan(object):

    def __init__(self, args, device, model_name='bert-base-uncased', dim=768):
        self.flamingo_model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="mosaicml/mpt-7b",
            tokenizer_path="mosaicml/mpt-7b",
            cross_attn_every_n_layers=4
        )
        self.flamingo_model.to(FLAMINGO_CONFIG['device'])
        self.flamingo_model.eval()
        
        self.eid2name, self.keywords, self.eid2idx = self.load_vocab(os.path.join(args.dataset, args.vocab))
        self.entity_pos = pickle.load(open(os.path.join(args.dataset, args.entity_pos_out), 'rb'))
        
        self.entity_images = self.load_entity_images(args.dataset)
        
        self.pretrained_emb = np.memmap(os.path.join(args.dataset, args.emb_file), 
                                        dtype='float32', mode='r', 
                                        shape=(self.entity_pos[-1], dim))
        self.means = np.array([np.mean(emb, axis=0) for emb in self.get_emb_iter()])
        self.inflect = inflect.engine()
        
        self.generation_templates = [
            "What do {}, {}, and {} have in common? They are all ",
            "{} , {} and {} are examples of ",
            "The common category for {}, {} and {} is ",
            "All of these - {}, {}, {} - are types of ",
            "{} , {} and {} belong to the category of "
        ]
        
        self.expansion_templates = [
            "Is {} a type of {}?",
            "Can {} be classified as {}?",
            "Would you categorize {} as {}?",
            "Is {} an example of {}?",
            "Does {} belong to the category of {}?"
        ]
        
        self.ranking_templates = [
            "How likely is it that {} is a type of {}?",
            "On a scale of 0 to 10, how relevant is {} to {}?",
            "Rate the relevance between {} and {} from 0 to 10.",
            "How strongly is {} associated with {}? (0-10)",
            "Does {} fit well in the category {}? (0-10)"
        ]
        
        self.calculated_cname_rep = {}
        self.k = args.k
        self.gen_thres = args.gen_thres
        self.device = device

    def load_vocab(self, vocab_path):
        eid2name = {}
        keywords = set()
        eid2idx = {}
        
        with open(vocab_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eid = int(parts[0])
                    name = parts[1]
                    eid2name[eid] = name
                    eid2idx[eid] = len(eid2idx)
                    keywords.add(name.lower())
        
        return eid2name, keywords, eid2idx

    def load_entity_images(self, dataset_path):
        image_mapping_path = os.path.join(dataset_path, "entity_images.pkl")
        if os.path.exists(image_mapping_path):
            return pickle.load(open(image_mapping_path, "rb"))
        
        entity_images = {}
        image_dir = os.path.join(dataset_path, "images")
        
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        for eid, name in self.eid2name.items():
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

    def get_entity_images(self, entity_ids):
        images = []
        for eid in entity_ids:
            img_path = self.entity_images.get(eid, "")
            if img_path and os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    image = self.image_processor(image).unsqueeze(0)
                    images.append(image)
                except Exception as e:
                    blank_image = Image.new('RGB', (224, 224), (128, 128, 128))
                    blank_image = self.image_processor(blank_image).unsqueeze(0)
                    images.append(blank_image)
        
        if images:
            return torch.cat(images, dim=0)
        return None

    def generate_with_flamingo(self, prompt, images=None, classification=False):
        vision_x = images.to(FLAMINGO_CONFIG['device']) if images is not None else None
        lang_x = self.tokenizer([prompt], return_tensors="pt").input_ids.to(FLAMINGO_CONFIG['device'])
        
        with torch.no_grad():
            outputs = self.flamingo_model.generate(
                vision_x=vision_x,
                lang_x=lang_x,
                max_new_tokens=FLAMINGO_CONFIG['max_new_tokens'],
                temperature=FLAMINGO_CONFIG['temperature'],
                num_beams=FLAMINGO_CONFIG['num_beams'],
                top_p=FLAMINGO_CONFIG['top_p'],
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(prompt, "").strip()
        
        if classification:
            generated_text = generated_text.lower()
            if "yes" in generated_text or "true" in generated_text or "correct" in generated_text:
                return "yes"
            elif "no" in generated_text or "false" in generated_text or "incorrect" in generated_text:
                return "no"
            else:
                return "no"
        else:
            words = nltk.word_tokenize(generated_text)
            pos_tags = nltk.pos_tag(words)
            
            noun_phrases = []
            current_phrase = []
            for word, pos in pos_tags:
                if pos.startswith('NN'):
                    current_phrase.append(word)
                elif current_phrase:
                    noun_phrases.append(" ".join(current_phrase))
                    current_phrase = []
            if current_phrase:
                noun_phrases.append(" ".join(current_phrase))
            
            return noun_phrases[0] if noun_phrases else generated_text.split('.')[0]

    def generate_ranking_score(self, prompt, images=None):
        vision_x = images.to(FLAMINGO_CONFIG['device']) if images is not None else None
        lang_x = self.tokenizer([prompt], return_tensors="pt").input_ids.to(FLAMINGO_CONFIG['device'])
        
        with torch.no_grad():
            outputs = self.flamingo_model.generate(
                vision_x=vision_x,
                lang_x=lang_x,
                max_new_tokens=3, 
                temperature=0.1,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(prompt, "").strip()
        
        try:
            for word in generated_text.split():
                if word.isdigit():
                    score = int(word)
                    if 0 <= score <= 10:
                        return score / 10.0  # 归一化到0-1
        except:
            pass
        
        return 0.5

    def rand_idx(self, n, start=0):
        indices = list(range(start, n))
        random.shuffle(indices)
        for i in indices:
            yield i

    def get_emb_iter(self):
        for i in range(len(self.entity_pos)):
            if i == 0:
                start, end = 0, self.entity_pos[i]
            else:
                start, end = self.entity_pos[i-1], self.entity_pos[i]
            yield self.pretrained_emb[start:end]

    def class_name_generation(self, entity_ids):
        cname2count = ddict(int)
        n = len(entity_ids)
        
        if n < 3:
            return cname2count
            
        idx_generator = self.rand_idx(n)
        
        for _ in range(min(GENERATION_SAMPLE_SIZE, n)):
            for template in self.generation_templates:
                indices = []
                for i in idx_generator:
                    if i not in indices:
                        indices.append(i)
                        if len(indices) == 3:
                            break
                
                selected_ids = [entity_ids[i] for i in indices]
                entity_names = [self.eid2name[eid] for eid in selected_ids]
                images = self.get_entity_images(selected_ids)
                
                prompt = template.format(*entity_names)
                
                try:
                    cname = self.generate_with_flamingo(prompt, images)
                except Exception as e:
                    print(f"Error generating category: {e}")
                    continue
                
                if cname:
                    singular_cname = self.inflect.singular_noun(cname)
                    if singular_cname:
                        cname = singular_cname
                    
                    cname2count[cname] += 1
        
        return cname2count

    def class_guided_expansion(self, cname, current_set, neg_set):
        current_ids = list(current_set)
        
        all_entity_ids = list(self.eid2name.keys())
        candidate_ids = [eid for eid in all_entity_ids 
                         if eid not in current_set and eid not in neg_set]
        
        random.shuffle(candidate_ids)
        candidate_ids = candidate_ids[:min(EXPANSION_SAMPLE_SIZE, len(candidate_ids))]
        
        entity_scores = ddict(int)
        
        for eid in candidate_ids:
            entity_name = self.eid2name[eid]
            images = self.get_entity_images([eid])
            
            for template in self.expansion_templates:
                prompt = template.format(entity_name, cname)
                
                try:
                    result = self.generate_with_flamingo(prompt, images, classification=True)
                except Exception as e:
                    print(f"Error classifying entity {entity_name}: {e}")
                    continue
                
                if result == "yes":
                    entity_scores[eid] += 1
        
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [eid for eid, score in sorted_entities if score >= FLAMINGO_CONFIG['threshold']]

    def class_name_ranking(self, cname2count, current_set, neg_set):
        cname_scores = []
        
        for cname, count in cname2count.items():
            total_score = 0
            sample_size = 0
            
            current_ids = list(current_set)
            random.shuffle(current_ids)
            sample_entities = current_ids[:min(5, len(current_ids))]
            
            if sample_entities:
                images = self.get_entity_images([sample_entities[0]])
            else:
                images = None
            
            for template in self.ranking_templates:
                for eid in sample_entities:
                    entity_name = self.eid2name[eid]
                    prompt = template.format(entity_name, cname)
                    
                    try:
                        score = self.generate_ranking_score(prompt, images)
                    except Exception as e:
                        print(f"Error ranking category {cname}: {e}")
                        continue
                        
                    total_score += score
                    sample_size += 1
            
            avg_score = total_score / sample_size if sample_size > 0 else 0
            cname_scores.append((cname, count, avg_score))
        
        cname_scores.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        return cname_scores

    def get_negative_samples(self, current_set, sample_size=20):
        current_embeddings = []
        for eid in current_set:
            if eid in self.eid2idx:
                idx = self.eid2idx[eid]
                if idx == 0:
                    emb = self.pretrained_emb[0:self.entity_pos[0]]
                else:
                    emb = self.pretrained_emb[self.entity_pos[idx-1]:self.entity_pos[idx]]
                current_embeddings.append(np.mean(emb, axis=0))
        
        if not current_embeddings:
            return []
        
        avg_embedding = np.mean(current_embeddings, axis=0)
        
        all_embeddings = []
        all_entity_ids = []
        for eid, idx in self.eid2idx.items():
            if eid not in current_set:
                if idx == 0:
                    emb = self.pretrained_emb[0:self.entity_pos[0]]
                else:
                    emb = self.pretrained_emb[self.entity_pos[idx-1]:self.entity_pos[idx]]
                all_embeddings.append(np.mean(emb, axis=0))
                all_entity_ids.append(eid)
        
        if not all_embeddings:
            return []
        
        similarities = cos([avg_embedding], all_embeddings)[0]
        
        sorted_indices = np.argsort(similarities)
        negative_samples = [all_entity_ids[i] for i in sorted_indices[:min(sample_size, len(sorted_indices))]]
        
        return negative_samples

    def expand(self, query_set, target_size, m=2, gt=None):
        current_set = set(query_set)
        expanded_set = set()
        neg_set = set()
        iteration = 0
        
        expand_path = []
        
        while len(current_set) < target_size:
            iteration += 1
            print(f"=== Iteration {iteration} ===")
            print(f"Current set size: {len(current_set)}")
            
            cname2count = self.class_name_generation(list(current_set))
            print(f"Generated {len(cname2count)} candidate categories")
            
            if not cname2count:
                print("No categories generated. Stopping expansion.")
                break
            
            cname_ranked = self.class_name_ranking(cname2count, current_set, neg_set)
            print(f"Top categories: {cname_ranked[:3]}")
            
            selected_cnames = [cname for cname, _, _ in cname_ranked[:m]]
            
            new_entities = set()
            for cname in selected_cnames:
                print(f"Expanding with category: {cname}")
                expanded = self.class_guided_expansion(cname, current_set, neg_set)
                print(f"Found {len(expanded)} entities for category {cname}")
                
                new_entities.update(expanded)
            
            if not new_entities:
                print("No new entities found. Stopping expansion.")
                break
            
            current_set.update(new_entities)
            expanded_set.update(new_entities)
            
            expand_path.append((selected_cnames, list(new_entities)))
            
            neg_samples = self.get_negative_samples(current_set)
            neg_set.update(neg_samples)
            print(f"Added {len(neg_samples)} negative samples")
            
            if len(current_set) >= target_size:
                print(f"Reached target size of {target_size}")
                break
        
        return list(current_set), expand_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='path to dataset folder')
    parser.add_argument('-vocab', default='entity2id.txt', help='vocab file')
    parser.add_argument('-sent', default='sentences.json', help='sent file')
    parser.add_argument('-emb_file', default='pretrained_emb.npy', help='name of pretrained embedding npy file')
    parser.add_argument('-entity_pos_out', default='entity_pos.pkl', help='name of entity index file')
    parser.add_argument('-output', default='results', help='file name for output')
    parser.add_argument('-k', default=5, help='k for soft match', type=int)
    parser.add_argument('-m', default=2, help='margin', type=int)
    parser.add_argument('-gen_thres', default=3, help='class name threshold', type=int)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    cgexpan = CGExpan(args, device)

    print("Starting entity expansion with OpenFlamingo")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Flamingo model: {FLAMINGO_CONFIG['model_name']}")

    output_dir = os.path.join(args.dataset, args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    query_dir = os.path.join(args.dataset, 'query')
    for file in tqdm(os.listdir(query_dir), desc="Processing query files"):
        query_sets = []
        with open(os.path.join(query_dir, file), encoding='utf-8') as f:
            for line in f:
                if line == 'EXIT\n': break
                temp = line.strip().split(' ')
                query_sets.append([int(eid) for eid in temp])
        
        gt = set()
        gt_file = os.path.join(args.dataset, 'gt', file)
        if os.path.exists(gt_file):
            with open(gt_file, encoding='utf-8') as f:
                for line in f:
                    temp = line.strip().split('\t')
                    eid = int(temp[0])
                    if int(temp[2]) >= 1:
                        gt.add(eid)
        else:
            print(f"Warning: Ground truth file not found for {file}")

        for i, query_set in enumerate(tqdm(query_sets, desc=f"Expanding queries in {file}")):
            query_names = [cgexpan.eid2name.get(eid, f"unknown_{eid}") for eid in query_set]
            print(f"\nExpanding query set {i}: {', '.join(query_names)}")
            
            expanded, expand_path = cgexpan.expand(query_set, 50, args.m, gt)
            
            output_file = os.path.join(output_dir, f'{i}_{file}')
            with open(output_file, 'w') as f:
                if gt:
                    apk10 = apk(gt, expanded, 10)
                    apk20 = apk(gt, expanded, 20)   
                    apk50 = apk(gt, expanded, 50)
                    print(f"AP@10: {apk10:.4f}, AP@20: {apk20:.4f}, AP@50: {apk50:.4f}", file=f)
                
                print("\n=== Expansion Path ===", file=f)
                for iter_idx, (categories, entities) in enumerate(expand_path):
                    entity_names = [cgexpan.eid2name.get(eid, f"unknown_{eid}") for eid in entities]
                    print(f"Iteration {iter_idx+1}:", file=f)
                    print(f"  Categories: {', '.join(categories)}", file=f)
                    print(f"  New entities: {', '.join(entity_names)}", file=f)
                
                print("\n=== Final Expanded Entities ===", file=f)
                for eid in expanded:
                    name = cgexpan.eid2name.get(eid, f"unknown_{eid}")
                    print(f"{eid}\t{name}", file=f)
            
            print(f"Results saved to {output_file}")

    print("Entity expansion completed!")
